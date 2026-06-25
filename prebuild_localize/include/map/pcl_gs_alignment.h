#pragma once

/**
 * @file pcl_gs_alignment.h
 * @brief Alignment pipeline for registering a filtered PCL point cloud to a
 *        3D Gaussian Splatting (3DGS) model.
 *
 * Pipeline (all functions are in namespace PclGsAlignment):
 *
 *   1.  loadPCLCloud()       – Load the filtered PCL via CloudReader (project-native).
 *   2.  loadGSCentroids()    – Load GS centroid positions via pcl::io::loadPLYFile.
 *   3.  loadGSData()         – Load full GS data (pos + scale + rotation) by manually
 *                              parsing the binary 3DGS PLY format.
 *   4.  alignPCLtoGS()       – Complete pipeline (calls steps 4a–4d internally):
 *       4a. voxelDownsample()  – Downsample both clouds for speed.
 *       4b. estimateNormals()  – OMP normal estimation on both clouds.
 *       4c. computeFPFH()      – OMP FPFH feature extraction.
 *       4d. roughAlignFPFH()   – FPFH + SampleConsensusPrerejective (RANSAC).
 *       4e. refineAlignGICP()  – GICP refinement; injects per-Gaussian covariances
 *                                (R * diag(exp(s)^2) * R^T) onto the target cloud when
 *                                gsData is provided.
 *
 * Returns: Eigen::Matrix4f   transform that maps points from PCL space → GS space.
 *
 * Usage example (see also the TODO block in src/map/main.cpp):
 *
 *   #include "map/pcl_gs_alignment.h"
 *   namespace Align = PclGsAlignment;
 *
 *   // --- load ---
 *   auto pclCloud = Align::loadPCLCloud(pclFile, cacheDir);
 *   auto gsCloud  = Align::loadGSCentroids(gsFolder);
 *   auto gsData   = Align::loadGSData(gsFolder);         // optional but recommended
 *
 *   // --- align ---
 *   Eigen::Matrix4d pcl2gs = Align::alignPCLtoGS(pclCloud, gsCloud, gsData);
 *
 *   // --- apply to a pose (world-to-camera, stored in PCL frame) ---
 *   Eigen::Matrix4d w2cam_gs = w2cam_pcl * pcl2gs.inverse();
 */

// ── PCL ──────────────────────────────────────────────────────────────────────
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>

// ── Project ──────────────────────────────────────────────────────────────────
#include "RTRenderer/cloudreader.h"
#include "RTRenderer/Octreegrid.h"

// ── Standard ─────────────────────────────────────────────────────────────────
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <memory>

namespace PclGsAlignment {

// ─── Type aliases ─────────────────────────────────────────────────────────────

using PointT   = pcl::PointXYZ;
using NormalT  = pcl::PointNormal;
using FeatureT = pcl::FPFHSignature33;

using CloudXYZ = pcl::PointCloud<PointT>;
using CloudN   = pcl::PointCloud<NormalT>;
using CloudF   = pcl::PointCloud<FeatureT>;

/// Convenience alias for GICP's covariance vector type (3×3 matrices).
using GicpT           = pcl::GeneralizedIterativeClosestPoint<PointT, PointT>;
using MatricesVector  = GicpT::MatricesVector;
using MatricesVectorPtr = GicpT::MatricesVectorPtr;

// ─── GaussianPoint struct ─────────────────────────────────────────────────────

/**
 * @brief A single 3DGS Gaussian: centroid position, log-scales and quaternion
 *        rotation as stored in the standard 3DGS PLY format.
 *
 * Call covariance() to obtain the anisotropic 3×3 covariance matrix
 *   C = R * diag(exp(s_i)^2) * R^T
 * that encodes the shape and orientation of the splat.
 */
struct GaussianPoint {
    float x{}, y{}, z{};
    float scale[3]{};   ///< log-scale values: exp(scale[i]) gives the actual extent
    float rot[4]{};     ///< quaternion stored as (w, x, y, z)

    Eigen::Matrix3d covariance() const {
        Eigen::Quaterniond q(
            static_cast<double>(rot[0]),  // w
            static_cast<double>(rot[1]),  // x
            static_cast<double>(rot[2]),  // y
            static_cast<double>(rot[3])   // z
        );
        q.normalize();
        Eigen::Matrix3d R = q.toRotationMatrix();
        const double s0 = std::exp(static_cast<double>(scale[0]));
        const double s1 = std::exp(static_cast<double>(scale[1]));
        const double s2 = std::exp(static_cast<double>(scale[2]));
        Eigen::DiagonalMatrix<double, 3> S2(s0 * s0, s1 * s1, s2 * s2);
        return R * S2 * R.transpose();
    }
};

// ─── Internal helpers ─────────────────────────────────────────────────────────

namespace detail {

/// Metadata about one PLY property (name, byte offset and size within a vertex row).
struct PlyPropInfo {
    std::string name;
    int offset{};   ///< byte offset inside one vertex record
    int size{};     ///< byte size (4 for float/int, 8 for double)
};

/// Parsed result from a PLY header that is enough to extract arbitrary float properties.
struct PlyHeader {
    bool        binaryLittleEndian{false};
    int         numVertices{0};
    int         vertexStride{0};
    std::vector<PlyPropInfo> props;
    std::streampos dataStart{};

    int offsetOf(const std::string& name) const noexcept {
        for (const auto& p : props)
            if (p.name == name) return p.offset;
        return -1;
    }
};

/**
 * @brief Parse a PLY header and return layout information.
 *        The stream @p f should be positioned at the start of the file;
 *        on return it is positioned at the first byte of binary vertex data.
 */
inline PlyHeader parsePlyHeader(std::ifstream& f) {
    PlyHeader hdr;
    std::string line;
    bool inVertex   = false;
    int  offset     = 0;

    while (std::getline(f, line)) {
        // Strip Windows-style CR
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line == "end_header") {
            hdr.dataStart     = f.tellg();
            hdr.vertexStride  = offset;
            break;
        }

        if (line.rfind("format binary_little_endian", 0) == 0) {
            hdr.binaryLittleEndian = true;
        } else if (line.rfind("element vertex", 0) == 0) {
            inVertex          = true;
            hdr.numVertices   = std::stoi(line.substr(15));
            offset            = 0;
        } else if (line.rfind("element", 0) == 0 && line.find("vertex") == std::string::npos) {
            inVertex = false;   // entering a different element (e.g. face)
        } else if (inVertex && line.rfind("property", 0) == 0) {
            std::istringstream ss(line);
            std::string tok_prop, tok_type, tok_name;
            ss >> tok_prop >> tok_type >> tok_name;
            const int sz = (tok_type == "double" || tok_type == "float64") ? 8 : 4;
            hdr.props.push_back({tok_name, offset, sz});
            offset += sz;
        }
    }
    return hdr;
}

} // namespace detail

// ─── Loading helpers ──────────────────────────────────────────────────────────

/**
 * @brief Convert an OctreeGrid block map (from CloudReader::loadCloud) into a
 *        pcl::PointCloud<pcl::PointXYZ>.
 *
 * Reuses the project-native loading path so the same octree-based pre-processing
 * (filtering, caching) is applied as everywhere else in the codebase.
 */
inline CloudXYZ::Ptr cloudFromGrid(const std::unordered_map<int, OctreeGrid::Block>& grid) {
    auto cloud = pcl::make_shared<CloudXYZ>();
    for (const auto& [key, block] : grid)
        for (const auto& pt : block.positions)
            cloud->emplace_back(pt.x, pt.y, pt.z);
    cloud->is_dense = false;
    return cloud;
}

/**
 * @brief Load a filtered PCL point cloud file (.ply / .e57 / …) using the
 *        project's CloudReader (identical to the rest of the codebase).
 *
 * @param pclFile   Absolute path to the point cloud file.
 * @param cacheDir  Cache directory forwarded to CloudReader (for octree caching).
 * @return pcl::PointCloud<pcl::PointXYZ>::Ptr
 */
inline CloudXYZ::Ptr loadPCLCloud(const std::filesystem::path& pclFile,
                                   const std::filesystem::path& cacheDir) {
    std::cout << "[Alignment] Loading PCL cloud: " << pclFile << std::endl;
    const auto grid  = CloudReader::loadCloud(pclFile, cacheDir);
    auto       cloud = cloudFromGrid(grid);
    std::cout << "[Alignment] PCL cloud: " << cloud->size() << " points." << std::endl;
    return cloud;
}

/**
 * @brief Load the centroid positions of a 3DGS model as a PCL point cloud.
 *
 * Automatically selects the highest-iteration point_cloud.ply inside
 * @p gsFolder/point_cloud/.
 *
 * @param gsFolder  Path to the Gaussian Splatting output directory.
 * @return pcl::PointCloud<pcl::PointXYZ>::Ptr  (xyz only, no auxiliary data)
 * @throws std::runtime_error if no point_cloud.ply is found or loading fails.
 */
inline CloudXYZ::Ptr loadGSCentroids(const std::filesystem::path& gsFolder, const int lod = 0) {
    // Locate highest-iteration PLY
    const std::filesystem::path pcDir = gsFolder / "point_cloud";
    int  bestIter = -1;
    std::filesystem::path bestPly;

    // (time < 18) ? "Good day." : "Good evening."
    std::string ply_file_name = "point_cloud" + ((lod != 0) ? ("_" + std::to_string(lod)) : "") + ".ply";

    for (const auto& entry : std::filesystem::directory_iterator(pcDir)) {
        if (!entry.is_directory()) continue;
        const std::string dirName = entry.path().filename().string();
        if (dirName.rfind("iteration_", 0) == 0) {
            const int iter = std::stoi(dirName.substr(10));
            if (iter > bestIter) {
                const std::filesystem::path candidate = entry.path() / ply_file_name;
                if (std::filesystem::exists(candidate)) {
                    bestIter = iter;
                    bestPly  = candidate;
                }
            }
        }
    }
    if (bestPly.empty())
        throw std::runtime_error("[Alignment] No point_cloud.ply found in: " + pcDir.string());

    std::cout << "[Alignment] Loading GS centroids: " << bestPly
              << "  (iteration " << bestIter << ")" << std::endl;

    auto cloud = pcl::make_shared<CloudXYZ>();
    if (pcl::io::loadPLYFile<PointT>(bestPly.string(), *cloud) < 0)
        throw std::runtime_error("[Alignment] pcl::io::loadPLYFile failed for: " + bestPly.string());

    std::cout << "[Alignment] GS centroids: " << cloud->size() << " points." << std::endl;
    return cloud;
}

/**
 * @brief Load full 3DGS data from the highest-iteration point_cloud.ply.
 *
 * Manually parses the binary-little-endian 3DGS PLY format to extract per-Gaussian
 * position (x, y, z), log-scales (scale_0..2) and quaternion rotation (rot_0..3).
 * The resulting GaussianPoint objects can be used to build per-point covariance
 * matrices for informed GICP registration.
 *
 * @param gsFolder  Path to the Gaussian Splatting output directory.
 * @return std::vector<GaussianPoint>
 * @throws std::runtime_error if the PLY is missing required properties or cannot
 *         be read.
 */
inline std::vector<GaussianPoint> loadGSData(const std::filesystem::path& gsFolder, const int lod = 0) {
    // Locate highest-iteration PLY (same logic as loadGSCentroids)
    std::string ply_file_name = "point_cloud" + ((lod != 0) ? ("_" + std::to_string(lod)) : "") + ".ply";

    const std::filesystem::path pcDir = gsFolder / ply_file_name;
    int  bestIter = -1;
    std::filesystem::path bestPly;

    for (const auto& entry : std::filesystem::directory_iterator(pcDir)) {
        if (!entry.is_directory()) continue;
        const std::string dirName = entry.path().filename().string();
        if (dirName.rfind("iteration_", 0) == 0) {
            const int iter = std::stoi(dirName.substr(10));
            if (iter > bestIter) {
                const std::filesystem::path candidate = entry.path() / "point_cloud.ply";
                if (std::filesystem::exists(candidate)) {
                    bestIter = iter;
                    bestPly  = candidate;
                }
            }
        }
    }
    if (bestPly.empty())
        throw std::runtime_error("[Alignment] No point_cloud.ply found in: " + pcDir.string());

    std::ifstream f(bestPly, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("[Alignment] Cannot open: " + bestPly.string());

    const auto hdr = detail::parsePlyHeader(f);

    if (!hdr.binaryLittleEndian)
        throw std::runtime_error("[Alignment] Only binary_little_endian PLY is supported "
                                 "for GS data. Got ASCII or big-endian format.");

    // Verify all required properties are present
    for (const char* prop : {"x","y","z","scale_0","scale_1","scale_2",
                              "rot_0","rot_1","rot_2","rot_3"}) {
        if (hdr.offsetOf(prop) < 0)
            throw std::runtime_error(std::string("[Alignment] GS PLY missing property: ") + prop);
    }

    std::vector<GaussianPoint> gaussians(static_cast<std::size_t>(hdr.numVertices));
    std::vector<char> row(static_cast<std::size_t>(hdr.vertexStride));

    f.seekg(hdr.dataStart);
    for (int i = 0; i < hdr.numVertices; ++i) {
        f.read(row.data(), hdr.vertexStride);
        if (!f)
            throw std::runtime_error("[Alignment] Unexpected EOF while reading GS PLY body.");

        const auto read_f = [&](const std::string& name) -> float {
            const int off = hdr.offsetOf(name);
            float v{};
            std::memcpy(&v, row.data() + off, sizeof(float));
            return v;
        };

        gaussians[i].x        = read_f("x");
        gaussians[i].y        = read_f("y");
        gaussians[i].z        = read_f("z");
        gaussians[i].scale[0] = read_f("scale_0");
        gaussians[i].scale[1] = read_f("scale_1");
        gaussians[i].scale[2] = read_f("scale_2");
        gaussians[i].rot[0]   = read_f("rot_0");   // w
        gaussians[i].rot[1]   = read_f("rot_1");   // x
        gaussians[i].rot[2]   = read_f("rot_2");   // y
        gaussians[i].rot[3]   = read_f("rot_3");   // z
    }

    std::cout << "[Alignment] Loaded " << gaussians.size()
              << " Gaussians (pos + scale + rot) from: " << bestPly << std::endl;
    return gaussians;
}

// ─── Preprocessing ────────────────────────────────────────────────────────────

/**
 * @brief Voxel-grid downsample a point cloud.
 *
 * @param cloud    Input cloud.
 * @param leafSize Voxel leaf size in metres (e.g. 0.2 for 20 cm voxels).
 */
inline CloudXYZ::Ptr voxelDownsample(const CloudXYZ::Ptr& cloud, float leafSize) {
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leafSize, leafSize, leafSize);
    auto filtered = pcl::make_shared<CloudXYZ>();
    vg.filter(*filtered);
    return filtered;
}

/**
 * @brief Estimate surface normals using OpenMP-accelerated PCA.
 *
 * @param cloud          Input XYZ cloud.
 * @param searchRadius   Neighbourhood radius for normal estimation (metres).
 * @return pcl::PointNormal cloud (positions + normals concatenated).
 */
inline CloudN::Ptr estimateNormals(const CloudXYZ::Ptr& cloud, float searchRadius = 0.5f) {
    pcl::PointCloud<pcl::Normal> rawNormals;

    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(searchRadius);
    ne.setSearchMethod(pcl::make_shared<pcl::search::KdTree<PointT>>());
    ne.compute(rawNormals);

    auto cloudWithNormals = pcl::make_shared<CloudN>();
    pcl::concatenateFields(*cloud, rawNormals, *cloudWithNormals);
    return cloudWithNormals;
}

// ─── Feature extraction ───────────────────────────────────────────────────────

/**
 * @brief Compute FPFH (Fast Point Feature Histograms) using OpenMP acceleration.
 *
 * @param cloudWithNormals  PointNormal cloud (output of estimateNormals()).
 * @param searchRadius      FPFH neighbourhood radius. Must be larger than the
 *                          radius used for normal estimation.
 * @return pcl::PointCloud<pcl::FPFHSignature33>::Ptr
 */
inline CloudF::Ptr computeFPFH(const CloudN::Ptr& cloudWithNormals, float searchRadius = 1.0f) {
    auto features = pcl::make_shared<CloudF>();

    pcl::FPFHEstimationOMP<NormalT, NormalT, FeatureT> fpfh;
    fpfh.setInputCloud(cloudWithNormals);
    fpfh.setInputNormals(cloudWithNormals);
    fpfh.setSearchMethod(pcl::make_shared<pcl::search::KdTree<NormalT>>());
    fpfh.setRadiusSearch(searchRadius);
    fpfh.compute(*features);

    return features;
}

// ─── Rough alignment: FPFH + RANSAC ──────────────────────────────────────────

/**
 * @brief Initial rough alignment via FPFH feature matching and
 *        SampleConsensusPrerejective (RANSAC with polygon similarity prerejection).
 *
 * Aligns @p sourceWithNormals (PCL) onto @p targetWithNormals (GS).
 *
 * @param sourceWithNormals  Source cloud with normals (PCL side).
 * @param targetWithNormals  Target cloud with normals (GS side).
 * @param sourceFPFH         FPFH features for source (from computeFPFH).
 * @param targetFPFH         FPFH features for target (from computeFPFH).
 * @param maxIterations      RANSAC iteration budget (default 50 000).
 * @param similarityThresh   Polygon similarity threshold 0–1 (higher = stricter
 *                           but faster; try 0.9).
 * @param maxCorrespondDist  Maximum Euclidean distance for a valid correspondence.
 * @param inlierFraction     Required inlier fraction for convergence (default 0.25).
 * @return 4×4 double transformation: source → target.
 * @throws std::runtime_error if RANSAC does not converge.
 */
inline Eigen::Matrix4d roughAlignFPFH(const CloudN::Ptr& sourceWithNormals,
                                       const CloudN::Ptr& targetWithNormals,
                                       const CloudF::Ptr& sourceFPFH,
                                       const CloudF::Ptr& targetFPFH,
                                       int   maxIterations     = 50000,
                                       float similarityThresh  = 0.9f,
                                       float maxCorrespondDist = 2.5f,
                                       float inlierFraction    = 0.25f) {
    pcl::SampleConsensusPrerejective<NormalT, NormalT, FeatureT> align;
    align.setInputSource(sourceWithNormals);
    align.setSourceFeatures(sourceFPFH);
    align.setInputTarget(targetWithNormals);
    align.setTargetFeatures(targetFPFH);
    align.setMaximumIterations(maxIterations);
    align.setNumberOfSamples(3);
    align.setCorrespondenceRandomness(5);
    align.setSimilarityThreshold(similarityThresh);
    align.setMaxCorrespondenceDistance(maxCorrespondDist);
    align.setInlierFraction(inlierFraction);

    CloudN aligned;
    align.align(aligned);

    if (!align.hasConverged())
        throw std::runtime_error("[Alignment] FPFH + RANSAC did not converge. "
                                 "Try increasing maxIterations or relaxing thresholds.");

    std::cout << "[Alignment] FPFH+RANSAC fitness score: "
              << align.getFitnessScore() << std::endl;
    return align.getFinalTransformation().cast<double>();
}

// ─── GICP covariance builder ──────────────────────────────────────────────────

/**
 * @brief Build a GICP-compatible covariance vector for a *downsampled* cloud.
 *
 * For each point in @p cloud a KNN lookup (k=1) is performed into the original
 * full-resolution GS centroid set, and the nearest Gaussian's covariance
 * C = R * diag(exp(s_i)^2) * R^T is assigned. This keeps the covariance count
 * equal to cloud->size() (required by setTargetCovariances) while avoiding
 * computing millions of matrix products for the full GS cloud.
 *
 * @param cloud    Downsampled GS centroid cloud (target cloud for GICP).
 * @param gsData   Full GS data (from loadGSData); used only as a covariance lookup.
 * @return MatricesVectorPtr – index-aligned to @p cloud.
 */
inline MatricesVectorPtr buildGSCovariancesForCloud(
        const CloudXYZ::Ptr&              cloud,
        const std::vector<GaussianPoint>& gsData) {

    // Build a KD-tree over the original full-res GS centroids for NN lookup
    auto origCloud = pcl::make_shared<CloudXYZ>();
    origCloud->resize(gsData.size());
    for (std::size_t i = 0; i < gsData.size(); ++i)
        (*origCloud)[i] = PointT(gsData[i].x, gsData[i].y, gsData[i].z);

    pcl::search::KdTree<PointT> tree;
    tree.setInputCloud(origCloud);

    auto covs = std::make_shared<MatricesVector>();
    covs->reserve(cloud->size());

    std::vector<int>   idx(1);
    std::vector<float> dist(1);
    for (const auto& pt : *cloud) {
        tree.nearestKSearch(pt, 1, idx, dist);
        covs->push_back(gsData[static_cast<std::size_t>(idx[0])].covariance());
    }
    return covs;
}

// ─── Refinement: GICP ────────────────────────────────────────────────────────

/**
 * @brief Refine an initial alignment using Generalised ICP.
 *
 * When @p gsCovariances is non-null it is injected as the target covariances so
 * that GICP accounts for the actual shape/scale of each Gaussian splat, which can
 * significantly improve convergence in areas with large, anisotropic splats.
 *
 * @param source             Source cloud – PCL points (xyz only).
 * @param target             Target cloud – GS centroids (xyz only).
 * @param initialGuess       4×4 rough transform from roughAlignFPFH().
 * @param gsCovariances      Optional per-Gaussian covariances (from buildGSCovariances).
 *                           Pass nullptr to let GICP compute its own covariances.
 * @param maxIterations      GICP iteration budget (default 200).
 * @param maxCorrespondDist  Maximum correspondence distance (metres, default 1.0).
 * @return 4×4 double refined transformation.
 * @throws std::runtime_error if GICP does not converge.
 */
inline Eigen::Matrix4d refineAlignGICP(const CloudXYZ::Ptr& source,
                                        const CloudXYZ::Ptr& target,
                                        const Eigen::Matrix4d& initialGuess,
                                        const MatricesVectorPtr& gsCovariances = nullptr,
                                        int   maxIterations     = 200,
                                        float maxCorrespondDist = 1.0f) {
    GicpT gicp;
    gicp.setInputSource(source);
    gicp.setInputTarget(target);
    gicp.setMaximumIterations(maxIterations);
    gicp.setMaxCorrespondenceDistance(maxCorrespondDist);
    gicp.setTransformationEpsilon(1e-6);
    gicp.setEuclideanFitnessEpsilon(1e-6);

    if (gsCovariances) {
        if (gsCovariances->size() == target->size()) {
            gicp.setTargetCovariances(gsCovariances);
        } else {
            std::cerr << "[Alignment] Warning: GS covariance count (" << gsCovariances->size()
                      << ") differs from target cloud size (" << target->size()
                      << "). Falling back to GICP auto-covariances.\n";
        }
    }

    CloudXYZ refined;
    gicp.align(refined, initialGuess.cast<float>());

    if (!gicp.hasConverged())
        throw std::runtime_error("[Alignment] GICP did not converge. "
                                 "Try a tighter roughAlignFPFH result or adjust maxCorrespondDist.");

    std::cout << "[Alignment] GICP fitness score: " << gicp.getFitnessScore() << std::endl;
    return gicp.getFinalTransformation().cast<double>();
}

// ─── Main pipeline ────────────────────────────────────────────────────────────

/**
 * @brief Full PCL → GS alignment pipeline.
 *
 * Steps:
 *   1. Voxel-downsample both input clouds (controls cost of ALL subsequent steps).
 *   2. Estimate surface normals on the downsampled clouds.
 *   3. Compute FPFH features on the normal-enriched clouds.
 *   4. Rough alignment: FPFH + SampleConsensusPrerejective (RANSAC).
 *   5. Refinement: GICP on the *same* downsampled clouds. When @p gsData is
 *      provided the per-Gaussian covariances are assigned to the downsampled
 *      GS cloud via nearest-neighbour lookup (one covariance per downsampled
 *      point, index-aligned, no full-cloud matrix loop).
 *
 * @param pclCloud    Filtered PCL cloud loaded via loadPCLCloud().
 * @param gsCloud     GS centroid cloud loaded via loadGSCentroids().
 * @param gsData      Full GS data from loadGSData(). Pass an empty vector to skip
 *                    Gaussian covariance injection (GICP will compute its own).
 * @param voxelLeaf   Voxel grid leaf size (metres). Controls density for all steps.
 *                    Default 0.2 m. Increase to 0.4–0.5 m for faster runs.
 * @param normalRadius Search radius for normal estimation (metres). Must be >
 *                    voxelLeaf. Default 0.5 m.
 * @param fpfhRadius  Search radius for FPFH features (metres, must exceed
 *                    normalRadius). Default 1.0 m.
 * @return Eigen::Matrix4d   transform: PCL coordinate frame → GS coordinate frame.
 * @throws std::runtime_error on loading or convergence failure.
 */
inline Eigen::Matrix4d alignPCLtoGS(const CloudXYZ::Ptr&              pclCloud,
                                     const CloudXYZ::Ptr&              gsCloud,
                                     const std::vector<GaussianPoint>& gsData      = {},
                                     float                             voxelLeaf   = 0.2f,
                                     float                             normalRadius = 0.5f,
                                     float                             fpfhRadius   = 1.0f) {
    std::cout << "\n=== PCL ↔ GS Alignment ===" << std::endl;

    // ── 1. Downsample for speed ──────────────────────────────────────────────
    //    pclDS / gsDS are reused for BOTH FPFH and GICP so the full-resolution
    //    clouds (potentially tens of millions of points) are never fed directly
    //    into any registration algorithm.
    auto pclDS = voxelDownsample(pclCloud, voxelLeaf);
    auto gsDS  = voxelDownsample(gsCloud,  voxelLeaf);
    std::cout << "[Alignment] After voxel downsampling: PCL=" << pclDS->size()
              << "  GS=" << gsDS->size() << std::endl;

    // ── 2. Normal estimation ─────────────────────────────────────────────────
    auto pclN = estimateNormals(pclDS, normalRadius);
    auto gsN  = estimateNormals(gsDS,  normalRadius);

    // ── 3. FPFH features ─────────────────────────────────────────────────────
    auto pclF = computeFPFH(pclN, fpfhRadius);
    auto gsF  = computeFPFH(gsN,  fpfhRadius);

    // ── 4. Rough alignment: FPFH + RANSAC ────────────────────────────────────
    std::cout << "[Alignment] Running FPFH + RANSAC rough alignment..." << std::endl;
    const Eigen::Matrix4d roughT = roughAlignFPFH(pclN, gsN, pclF, gsF);

    // ── 5. GICP refinement on the downsampled clouds ─────────────────────────
    //    Source: pclDS (already downsampled above)
    //    Target: gsDS  (already downsampled above)
    //    Covariances: assigned per downsampled GS point via NN lookup into the
    //    original gsData, so we only compute O(|gsDS|) covariances, not O(|gsData|).
    std::cout << "[Alignment] Running GICP refinement..." << std::endl;

    MatricesVectorPtr gsCovs;
    if (!gsData.empty()) {
        std::cout << "[Alignment] Building per-Gaussian covariances for "
                  << gsDS->size() << " downsampled GS points (NN lookup)..." << std::endl;
        gsCovs = buildGSCovariancesForCloud(gsDS, gsData);
    }

    const Eigen::Matrix4d finalT = refineAlignGICP(pclDS, gsDS, roughT, gsCovs);

    std::cout << "[Alignment] Final transform (PCL → GS):\n" << finalT << "\n" << std::endl;
    return finalT;
}

} // namespace PclGsAlignment
