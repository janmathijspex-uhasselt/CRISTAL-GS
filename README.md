# CRISTAL: Real-time Camera Registration in Static LiDAR Scans using Neural Rendering
This is the implementation of the arxiv publication: [CRISTAL: Real-time Camera Registration in Static LiDAR Scans using Neural Rendering](https://doi.org/10.48550/arXiv.2511.16349)

*Accurate camera localization is crucial for robotics and Extended Reality (XR), enabling reliable navigation and alignment of virtual and real content. Existing visual methods often suffer from drift, scale ambiguity, and depend on fiducials or loop closure. This work introduces a real-time method for localizing a camera within a pre-captured, highly accurate colored LiDAR point cloud. By rendering synthetic views from this cloud, 2D-3D correspondences are established between live frames and the point cloud. A neural rendering technique narrows the domain gap between synthetic and real images, reducing occlusion and background artifacts to improve feature matching. The result is drift-free camera tracking with correct metric scale in the global LiDAR coordinate system. Two real-time variants are presented: Online Render and Match, and Prebuild and Localize. We demonstrate improved results on the ScanNet++ dataset and outperform existing SLAM pipelines.*

## Citation
```
@misc{vanherck2025cristalrealtimecameraregistration,
      title={CRISTAL: Real-time Camera Registration in Static LiDAR Scans using Neural Rendering}, 
      author={Joni Vanherck and Steven Moonen and Brent Zoomers and Kobe Werner and Jeroen Put and Lode Jorissen and Nick Michiels},
      year={2025},
      eprint={2511.16349},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16349}, 
}
```

## Installation
Prerequisites:
- [Stella_vSLAM with Iridescence viewer](https://stella-cv.readthedocs.io/en/latest/overview.html#installation)
- [RTRenderer](https://github.com/EDM-Research/Real-time-Neural-Rendering-of-LiDAR-Point-Clouds.git)

```
git clone --recurse-submodules https://github.com/JVanherck/CRISTAL.git
cd CRISTAL
mkdir build && cd build
cmake ..
make
```
Optionally install with `sudo make install`

## Usage

### Render and match

`render_match <folder> <pcl_name> <video_name> <calibration_file>`
- folder: should point to the folder containing all data. The estimated trajectory and map will also be written to this directory.
- pcl_name: filename of the point cloud relative to \<folder\> (should be .ply or .e57)
- video_name: filename of the video to localize relative to \<folder\>
- calibration_file: filename of the calibration file relative to \<folder\>.
    - if named cameras.txt, COLMAP output is expected: `CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]`
    - in other cases, this format is expected:
```
        W H
        fx 0 cx
        0 fy cy
        0 0 1
        k1 k2 p1 p2 k3
        0
```

### Prebuild and Localize

First create a SLAM map offline:

`prebuild_map <folder> <cloud> <calib>`

- folder: should point to the folder containing all data. The map will also be written to this directory.
- cloud: filename of the point cloud relative to \<folder\> (should be .ply or .e57)
- calib: filename of the calibration file relative to \<folder\>. The same formats are expected as above. 

Then, localize the video:

`localize_video <orb_voc_path> <stella_config_path> <folder> <video_name> <calib_name> [map_name]`

- orb_voc_path: absolute path to the orb vocabulary file (stella_config/orb_vocab.fbow)
- stella_config_path: absolute path to the stella config file (stella_config/orb_vocab.fbow)
- folder: should point to the folder containing all data. The estimated trajectory will also be written to this directory.
- video_name: filename of the video to localize relative to \<folder\>
- calib_name: filename of the calibration file relative to \<folder\>. The same formats are expected as above.
- map_name: (Optional) name of the map file (stored in `<folder>/.cache`). If created by prebuild_map, the name will be prebuild.msg. If omitted, a new map will be constructed called default.msg using Stella_vSLAM.

### Render trajectory

To empirically see the accuracy of the localization, you can run the following command with the estimated trajectory file:

`render_trajectory <folder> <cloud> <video_name> <calibration_file> <trajectory_file>`

- folder: should point to the folder containing all data. A new folder will be created with side-by-side comparisons and blended images.
- cloud: filename of the point cloud relative \<folder\> (should be .ply or .e57)
- video_name: filename of the video that was localized relative to \<folder\>
- calibration_file: filename of the calibration file relative to \<folder\>. The same formats are expected as above.
- trajectory_file: filename of the estimated trajectory relative to \<folder\>. Expected format is TUM.

## About
![](/img/DFL_FlandersMake.jpg)

This work is a development by [Hasselt University](https://www.uhasselt.be/), [Digital Future Lab](https://www.uhasselt.be/en/instituten-en/digitalfuturelab), funded by Hasselt University and Research Foundation Flanders (FWO).
