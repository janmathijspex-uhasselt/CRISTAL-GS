#include "common/pose_generator.h"
#include "common/pose_config.h"
#include "common/geometry.h"
#include <limits>
#include <unordered_map>
#include <random>

using namespace cv;

// ── internal helpers ────────────────────────────────────────────────

struct SliceParams {
    int sliceMin = 0;
    int sliceMax = 0;
    int trackbarMax = 1000;
    float axisMinValue = 0.0f;
    int trackbarResolution = CONFIG_TRACKBAR_RES;
    std::string projectionPlane = "XY";
    const std::unordered_map<int, OctreeGrid::Block>* grid = nullptr;
    std::string windowName = CONFIG_WINDOW_NAME;
    float scale = 1.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    float minX = 0.0f;
    float minY = 0.0f;
};

static SliceParams params;
static float currentSampleSpacing = 0.2;
static int sampleSliderValue = 200; // represents 0.2 initially (scaled by ×100)
static float dilateDistance = CONFIG_DILATE_DISTANCE;
static int width = CONFIG_WINDOW_WIDTH, height = CONFIG_WINDOW_HEIGHT;
static cv::Mat display = cv::Mat::zeros(height, width, CV_8UC3);
static cv::Mat fillMask;
static bool isDrawing = false;
static cv::Point lastDrawnPoint(-1, -1);
static cv::Mat globalMask;  // accumulated flood-fill regions (blue channel)
static cv::Mat globalEdges; // edges computed from globalMask (green channel)

static cv::Point2f projectPoint(const cv::Point3f& pt, const std::string& plane) {
    if (plane == "XY")
        return {pt.x, pt.y};
    else if (plane == "XZ")
        return {pt.x, pt.z};
    else // YZ
        return {pt.y, pt.z};
}

static float getOrthogonalAxis(const cv::Point3f& pt, const std::string& plane) {
    if (plane == "XY") return pt.z;
    else if (plane == "XZ") return pt.y;
    else return pt.x;
}

static void computeScaling()
{
    const auto& grid = *params.grid;

    float axisMin = params.axisMinValue;
    float minAxis = axisMin + (params.sliceMin / static_cast<float>(params.trackbarResolution));
    float maxAxis = axisMin + (params.sliceMax / static_cast<float>(params.trackbarResolution));

    float minX = std::numeric_limits<float>::max(), minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest(), maxY = std::numeric_limits<float>::lowest();

    for (const auto& pair : grid) {
        for (const auto& pt : pair.second.positions) {
            float axis = getOrthogonalAxis(pt, params.projectionPlane);
            if (axis < minAxis || axis > maxAxis) continue;
            auto p = projectPoint(pt, params.projectionPlane);
            minX = std::min(minX, p.x);
            minY = std::min(minY, p.y);
            maxX = std::max(maxX, p.x);
            maxY = std::max(maxY, p.y);
        }
    }

    float margin = 0.05f * std::max(width, height);
    float rangeX = maxX - minX + 1e-5f;
    float rangeY = maxY - minY + 1e-5f;
    params.minX = minX;
    params.minY = minY;
    params.scale  = std::min((width - 2 * margin) / rangeX, (height - 2 * margin) / rangeY);

    params.offsetX = (width - params.scale * rangeX) / 2.0f;
    params.offsetY = (height - params.scale * rangeY) / 2.0f;
}

static void paintCallback(int event, int x, int y, int flags, void*) {
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        isDrawing = true;
        lastDrawnPoint = cv::Point(x, y);
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        isDrawing = false;
        lastDrawnPoint = cv::Point(-1, -1);
    }
    else if (event == cv::EVENT_MOUSEMOVE && isDrawing)
    {
        if (x >= 0 && x < display.cols && y >= 0 && y < display.rows)
        {
            cv::Point currentPoint(x, y);
            if (lastDrawnPoint.x >= 0 && lastDrawnPoint.y >= 0)
            {
                std::vector<cv::Mat> channels(3);
                cv::split(display, channels);
                cv::line(channels[2], lastDrawnPoint, currentPoint, 255, 2);
                cv::merge(channels, display);
            }
            lastDrawnPoint = currentPoint;
            cv::imshow(params.windowName, display);
        }
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        if (x >= 0 && x < display.cols && y >= 0 && y < display.rows && display.at<cv::Vec3b>(y,x) == cv::Vec3b{0,0,0})
        {
            std::vector<cv::Mat> channels(3);
            cv::split(display, channels);

            cv::Scalar floodColor = cv::Scalar(255, 0, 0);
            cv::Scalar loDiff = cv::Scalar(5, 5, 5);
            cv::Scalar upDiff = cv::Scalar(5, 5, 5);
            int flags = 4 | (255 << 8);
            cv::floodFill(display, cv::Point(x, y), floodColor, 0, loDiff, upDiff, flags);

            cv::split(display, channels);

            int iterations = std::max(1, static_cast<int>(dilateDistance * params.scale));
            cv::erode(channels[0], channels[0], cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), iterations);

            cv::Mat edges;
            cv::Canny(channels[1], edges, 100,110);
            globalEdges = edges;
            cv::imshow(params.windowName, display);
        }
    }
}

static std::vector<cv::Point2f> sampleScanPositions(const cv::Mat &mask, double distance)
{
    std::vector<cv::Point2f> scanPositions;
    for(int x = (int) distance/2; x < mask.cols; x += distance)
    {
        for(int y = (int) distance/2; y < mask.rows; y += distance)
        {
            if(mask.at<uchar>(y,x) > 0)
            {
                scanPositions.push_back(cv::Point2f{(float)x, (float)y});
            }
        }
    }
    return scanPositions;
}

static void updateDisplay(int, void*) {
    const auto& grid = *params.grid;

    display = cv::Mat::zeros(height, width, CV_8UC3);

    float axisMin = params.axisMinValue;
    float minAxis = axisMin + (params.sliceMin / static_cast<float>(params.trackbarResolution));
    float maxAxis = axisMin + (params.sliceMax / static_cast<float>(params.trackbarResolution));

    for (const auto& pair : grid) {
        for (const auto& pt : pair.second.positions) {
            float axis = getOrthogonalAxis(pt, params.projectionPlane);
            if (axis < minAxis || axis > maxAxis) continue;

            auto proj = projectPoint(pt, params.projectionPlane);
            int x = static_cast<int>((proj.x - params.minX) * params.scale + params.offsetX);
            int y = static_cast<int>((proj.y - params.minY) * params.scale + params.offsetY);

            circle(display, Point(x, height - y), 1, Scalar(0, 255, 0), -1);
        }
    }

    cv::Mat displayClone = display.clone();
    char buf[100];
    snprintf(buf, sizeof(buf), "Slice range: %.2f - %.2f", minAxis, maxAxis);
    putText(displayClone, buf, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));

    imshow(params.windowName, displayClone);
}

static void updateSampleVisualization(int, void*) {
    currentSampleSpacing = sampleSliderValue / 100.0f;

    std::vector<cv::Mat> channels(3);
    cv::split(display, channels);
    cv::Mat blueMask = channels[0] > 0;

    std::vector<cv::Point2f> scanPositions = sampleScanPositions(blueMask, currentSampleSpacing * params.scale);

    cv::Mat vis = display.clone();
    for (auto& p : scanPositions) {
        cv::circle(vis, cv::Point((int)p.x, (int)p.y), 3, cv::Scalar(255, 255, 255), -1);
    }

    char buf[100];
    snprintf(buf, sizeof(buf), "Sample spacing: %.2f m, #positions: %d", currentSampleSpacing, (int)scanPositions.size());
    putText(vis, buf, Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));

    imshow(params.windowName, vis);
}

static std::vector<cv::Point3f> unprojectTo3D(const std::vector<cv::Point2f>& scanPositions2D, const SliceParams& params)
{
    std::vector<cv::Point3f> result;
    float minAxis = params.axisMinValue + (params.sliceMin / static_cast<float>(params.trackbarResolution));
    float maxAxis = params.axisMinValue + (params.sliceMax / static_cast<float>(params.trackbarResolution));
    float midAxis = maxAxis;

    for (const auto& pt : scanPositions2D)
    {
        float x_img = pt.x;
        float y_img = height - pt.y;

        float x_proj = (x_img - params.offsetX) / params.scale + params.minX;
        float y_proj = (y_img - params.offsetY) / params.scale + params.minY;

        if (params.projectionPlane == "XY") {
            result.emplace_back(cv::Point3f(x_proj, y_proj, midAxis));
        } else if (params.projectionPlane == "XZ") {
            result.emplace_back(cv::Point3f(x_proj, midAxis, y_proj));
        } else if (params.projectionPlane == "YZ") {
            result.emplace_back(cv::Point3f(midAxis, x_proj, y_proj));
        }
    }

    return result;
}

static cv::Matx33d randomRotationMatrixAroundZ()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> angle_dist(CONFIG_RANDOM_ROTATION_MIN, CONFIG_RANDOM_ROTATION_MAX);
    double theta = angle_dist(gen);

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    cv::Matx33d R;
    R(0, 0) = cos_theta;  R(0, 1) = -sin_theta; R(0, 2) = 0.0;
    R(1, 0) = sin_theta;  R(1, 1) =  cos_theta; R(1, 2) = 0.0;
    R(2, 0) = 0.0;        R(2, 1) = 0.0;        R(2, 2) = 1.0;

    return R;
}

struct Face {
    cv::Vec3d direction;
    cv::Vec3d up;
};

static std::vector<Face> generateTiltedFaces(
    const std::vector<Face>& baseFaces,
    const std::vector<double>& tiltAnglesDeg
) {
    std::vector<Face> result;
    result.reserve(baseFaces.size() * tiltAnglesDeg.size());

    for (const auto& face : baseFaces) {
        cv::Vec3d d = cv::normalize(face.direction);
        cv::Vec3d u = cv::normalize(face.up);
        cv::Vec3d right = cv::normalize(d.cross(u));

        for (double tiltDeg : tiltAnglesDeg) {
            double tiltRad = tiltDeg * CV_PI / 180.0;
            double cosT = std::cos(tiltRad);
            double sinT = std::sin(tiltRad);

            cv::Vec3d tilted = cv::normalize(d * cosT + right.cross(d) * sinT);

            result.push_back({tilted, u});
        }
    }

    return result;
}

static std::vector<cv::Matx44d> positionToPose(const std::vector<cv::Point3f>& positions)
{
    std::vector<Face> baseFaces = {
        {{ 1,  0,  0}, {0, 0, -1}},
        {{-1,  0,  0}, {0, 0, -1}},
        {{ 0,  1,  0}, {0, 0, -1}},
        {{ 0, -1,  0}, {0, 0, -1}},
    };

    std::vector<double> tiltAngles = CONFIG_TILT_ANGLES_DEG;
    std::vector<Face> faces = generateTiltedFaces(baseFaces, tiltAngles);

    std::vector<cv::Matx44d> poses;

    for (const auto& pos : positions)
    {
        cv::Vec3d posvec(pos.x, pos.y, pos.z);
        cv::Matx33d R = randomRotationMatrixAroundZ();

        for (const auto& face : faces)
        {
            cv::Vec3d dir = R * face.direction;
            cv::Vec3d up = R * face.up;

            cv::Vec3d target = posvec + dir;
            cv::Matx44d w2cam = lookAt(posvec, target, up);
            poses.push_back(w2cam);
        }
    }

    return poses;
}

// ── shared interactive GUI for collecting scan positions ────────────

static std::vector<cv::Point3f> collectPositionsFromGUI(std::unordered_map<int, OctreeGrid::Block> grid)
{
    OctreeGrid::downSample(grid, 0.1);
    params.grid = &grid;

    auto recomputeSliceRange = [&]() {
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        for (const auto& pair : grid) {
            for (const auto& pt : pair.second.positions) {
                float axis = getOrthogonalAxis(pt, params.projectionPlane);
                minVal = std::min(minVal, axis);
                maxVal = std::max(maxVal, axis);
            }
        }

        params.axisMinValue = minVal;

        float range = maxVal - minVal;
        params.trackbarMax = std::max(1, static_cast<int>(range * params.trackbarResolution));
        params.sliceMin = 0;
        params.sliceMax = params.trackbarMax;

        // Update trackbars
        setTrackbarMax("Slice min", params.windowName, params.trackbarMax);
        setTrackbarMax("Slice max", params.windowName, params.trackbarMax);
        // Set initial float values
        float sliceMinFloat = CONFIG_SLICE_MIN_FLOAT;
        float sliceMaxFloat = CONFIG_SLICE_MAX_FLOAT;

        params.sliceMin = static_cast<int>((sliceMinFloat - params.axisMinValue) * params.trackbarResolution);
        params.sliceMax = static_cast<int>((sliceMaxFloat - params.axisMinValue) * params.trackbarResolution);

        // Clamp to valid range
        params.sliceMin = std::clamp(params.sliceMin, 0, params.trackbarMax);
        params.sliceMax = std::clamp(params.sliceMax, 0, params.trackbarMax);

        // Update trackbars
        setTrackbarPos("Slice min", params.windowName, params.sliceMin);
        setTrackbarPos("Slice max", params.windowName, params.sliceMax);

        computeScaling();
    };

    namedWindow(params.windowName);
    createTrackbar("Slice min", params.windowName, &params.sliceMin, params.trackbarMax, updateDisplay);
    createTrackbar("Slice max", params.windowName, &params.sliceMax, params.trackbarMax, updateDisplay);
    setMouseCallback(params.windowName, paintCallback);
    // Sample spacing (0.2 m to 8 m, mapped as 200–8000)
    createTrackbar("Sample spacing (×0.01m)", params.windowName, &sampleSliderValue, 8000, updateSampleVisualization);
    setTrackbarMin("Sample spacing (×0.01m)", params.windowName, 20); // min 0.2m
    setTrackbarPos("Sample spacing (×0.01m)", params.windowName, 200);

    recomputeSliceRange();
    updateDisplay(0, nullptr);

    struct SliceData {
        cv::Mat mask;
        float sampleSpacing;
    };
    std::unordered_map<int, SliceData> sliceDataMap;

    while (true) {
        char key = waitKey(0);
        if (key == 27) {  // ESC to quit
            destroyAllWindows();
            exit(0);
        }

        bool changedPlane = false;
        if (key == '1') {
            params.projectionPlane = "XZ";
            changedPlane = true;
        } else if (key == '2') {
            params.projectionPlane = "XY";
            changedPlane = true;
        } else if (key == '3') {
            params.projectionPlane = "YZ";
            changedPlane = true;
        }
        if(key == 'd')
        {
            cv::dilate(display, display, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), dilateDistance * params.scale);
            imshow(params.windowName, display);
            continue;
        }

        if (changedPlane) {
            recomputeSliceRange();
            updateDisplay(0, nullptr);
        }

        if(key == 'r')
        {
            updateDisplay(0, nullptr);
        }
        if (key == 's') {
            // Extract the flood-filled mask (blue channel)
            std::vector<cv::Mat> channels(3);
            cv::split(display, channels);
            cv::Mat blueMask = channels[0] > 0;

            // Save mask and current sample spacing for this slice
            SliceData data;
            data.mask = blueMask.clone();
            data.sampleSpacing = currentSampleSpacing;
            sliceDataMap[params.sliceMax] = std::move(data);

            std::cout << "Saved mask for slice " << params.sliceMax
                      << " (" << params.projectionPlane << "), spacing=" << currentSampleSpacing << " m\n";

            // After saving mask
            updateSampleVisualization(0, nullptr);

            continue;
        }
        if (key == 'q') {
            cv::destroyAllWindows();
            break;
        }
    }

    std::vector<cv::Point3f> allPositions3d;

    for (const auto& [sliceIdx, data] : sliceDataMap)
    {
        const auto& mask = data.mask;

        int oldMin = params.sliceMin, oldMax = params.sliceMax;
        params.sliceMin = sliceIdx;
        params.sliceMax = sliceIdx; // thin slice

        // Use the stored per-slice sample spacing
        std::vector<cv::Point2f> scanPositions =
            sampleScanPositions(mask, data.sampleSpacing * params.scale);

        std::vector<cv::Point3f> positions3d = unprojectTo3D(scanPositions, params);

        allPositions3d.insert(allPositions3d.end(), positions3d.begin(), positions3d.end());

        params.sliceMin = oldMin;
        params.sliceMax = oldMax;
    }

    return allPositions3d;
}

// ── public API ──────────────────────────────────────────────────────

namespace PoseGenerator {

std::vector<cv::Point3f> getPositions(std::unordered_map<int, OctreeGrid::Block> grid)
{
    return collectPositionsFromGUI(std::move(grid));
}

std::vector<cv::Matx44d> getPoses(std::unordered_map<int, OctreeGrid::Block> grid)
{
    auto positions = collectPositionsFromGUI(std::move(grid));
    return positionToPose(positions);
}

}
