#include "gpu_context.h"
#include "image_preprocessor.h"
#include "apriltag_gpu.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PyCudaAprilTag {
public:
    PyCudaAprilTag(int width,
                   int height,
                   int decimation,
                   float fx, float fy, float cx, float cy,
                   float tag_size_m)
        : ctx_(0),
          pre_(ctx_, width, height, decimation,
               nullptr) {
        cv::Matx33f K(fx, 0.f, cx,
                      0.f, fy, cy,
                      0.f, 0.f, 1.f);
        det_ = std::make_unique<AprilTagGpuDetector>(
            ctx_, pre_.workingWidth(), pre_.workingHeight(),
            tag_size_m, K);
    }

    py::list detect(py::array_t<uint8_t> frame) {
        py::buffer_info info = frame.request();
        if (info.ndim != 2 && info.ndim != 3) {
            throw std::runtime_error("Expected 2D (gray) or 3D (BGR) array");
        }
        int h = static_cast<int>(info.shape[0]);
        int w = static_cast<int>(info.shape[1]);

        cv::Mat cv_frame;
        if (info.ndim == 2) {
            cv_frame = cv::Mat(h, w, CV_8UC1, info.ptr);
        } else {
            int c = static_cast<int>(info.shape[2]);
            if (c != 3) {
                throw std::runtime_error("Expected 3-channel BGR image");
            }
            cv_frame = cv::Mat(h, w, CV_8UC3, info.ptr);
        }

        unsigned char* d_gray = pre_.preprocess(cv_frame);
        auto dets = det_->detect(d_gray);

        py::list out;
        for (const auto& d : dets) {
            py::dict item;
            item["id"] = d.id;
            py::list corners;
            for (int i = 0; i < 4; ++i) {
                corners.append(py::make_tuple(d.corners[i].x, d.corners[i].y));
            }
            item["corners"] = corners;
            out.append(item);
        }
        return out;
    }

private:
    GpuContext ctx_;
    ImagePreprocessor pre_;
    std::unique_ptr<AprilTagGpuDetector> det_;
};

PYBIND11_MODULE(cuda_apriltag_py, m) {
    py::class_<PyCudaAprilTag>(m, "CudaAprilTag")
        .def(py::init<int, int, int, float, float, float, float, float>(),
             py::arg("width"),
             py::arg("height"),
             py::arg("decimation") = 2,
             py::arg("fx") = 1000.f,
             py::arg("fy") = 1000.f,
             py::arg("cx") = 640.f,
             py::arg("cy") = 360.f,
             py::arg("tag_size_m") = 0.165f)
        .def("detect", &PyCudaAprilTag::detect,
             py::arg("frame"));
}


