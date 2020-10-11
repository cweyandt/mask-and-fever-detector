.ONESHELL:
.PHONY: deps download build clean

# CPU architecture


# OpenCV version to use.
OPENCV_VERSION?=4.4.0

# Temporary directory to put files into.
TMP_DIR?=/tmp/

# Build shared or static library
BUILD_SHARED_LIBS?=ON

# Package list for each well-known Linux distribution
RPMS=cmake curl wget git gtk2-devel libpng-devel libjpeg-devel libtiff-devel tbb tbb-devel libdc1394-devel unzip
DEBS=unzip wget build-essential cmake curl git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

# Detect Linux distribution
distro_deps=
ifneq ($(shell which dnf 2>/dev/null),)
	distro_deps=deps_fedora
else
ifneq ($(shell which apt-get 2>/dev/null),)
	distro_deps=deps_debian
else
ifneq ($(shell which yum 2>/dev/null),)
	distro_deps=deps_rh_centos
endif
endif
endif

SUDO=
ifneq ($(shell whoami),root)
	SUDO=sudo 
endif

# Install all necessary dependencies.
deps: $(distro_deps)

deps_rh_centos:
	$(SUDO) yum -y install pkgconfig $(RPMS)

deps_fedora:
	$(SUDO) dnf -y install pkgconf-pkg-config $(RPMS)

deps_debian:
	$(SUDO) apt-get -y update
	$(SUDO) apt-get -y install $(DEBS)


# Download OpenCV source tarballs.
download:
	rm -rf $(TMP_DIR)opencv
	mkdir $(TMP_DIR)opencv
	cd $(TMP_DIR)opencv
	curl -Lo opencv.zip https://github.com/opencv/opencv/archive/$(OPENCV_VERSION).zip
	unzip -q opencv.zip
	curl -Lo opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$(OPENCV_VERSION).zip
	unzip -q opencv_contrib.zip
	rm opencv.zip opencv_contrib.zip
	cd -

# Build OpenCV.
build:
	cd $(TMP_DIR)opencv/opencv-$(OPENCV_VERSION)
	mkdir build
	cd build
	rm -rf *
	cmake \
		-D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
		-D OPENCV_EXTRA_MODULES_PATH=$(TMP_DIR)opencv/opencv_contrib-$(OPENCV_VERSION)/modules \
		-D BUILD_DOCS=OFF \
		-D BUILD_EXAMPLES=OFF \
		-D BUILD_TESTS=OFF \
		-D BUILD_PERF_TESTS=OFF \
		-D BUILD_opencv_java=NO \
		-D BUILD_opencv_python=YES \
		-D BUILD_opencv_python2=NO \
		-D BUILD_opencv_python3=YES \
		-D WITH_JASPER=OFF \
		-D OPENCV_GENERATE_PKGCONFIG=ON ..
	$(MAKE) -j $(shell nproc --all)
	$(MAKE) preinstall
	cd -

# Build OpenCV with cuda.
build_cuda:
	cd $(TMP_DIR)opencv/opencv-$(OPENCV_VERSION)
	mkdir build
	cd build
	rm -rf *
	cmake -j $(shell nproc --all) \
		-D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
		-D OPENCV_EXTRA_MODULES_PATH=$(TMP_DIR)opencv/opencv_contrib-$(OPENCV_VERSION)/modules \
		-D BUILD_DOCS=OFF \
		-D BUILD_EXAMPLES=OFF \
		-D BUILD_TESTS=OFF \
		-D BUILD_PERF_TESTS=OFF \
		-D BUILD_opencv_java=NO \
		-D BUILD_opencv_python=YES \
		-D BUILD_opencv_python2=NO \
		-D BUILD_opencv_python3=YES \
		-D WITH_JASPER=OFF \
		-D OPENCV_GENERATE_PKGCONFIG=ON \
		-D WITH_CUDA=ON \
		-D ENABLE_FAST_MATH=1 \
		-D CUDA_FAST_MATH=1 \
		-D WITH_CUBLAS=1 \
		-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
		-D BUILD_opencv_cudacodec=OFF \
		-D WITH_CUDNN=ON \
		-D OPENCV_DNN_CUDA=ON \
		-D CUDA_GENERATION=Auto ..
	$(MAKE) -j $(shell nproc --all)
	$(MAKE) preinstall
	cd -

# Cleanup temporary build files.
clean:
	rm -rf $(TMP_DIR)opencv

# Cleanup old library files.
sudo_pre_install_clean:
	$(SUDO) rm -rf /usr/local/lib/cmake/opencv4/
	$(SUDO) rm -rf /usr/local/lib/libopencv*
	$(SUDO) rm -rf /usr/local/lib/pkgconfig/opencv*
	$(SUDO) rm -rf /usr/local/include/opencv*

# Do everything.
install:
ifeq ($(ARCH),aarch64)
	echo running cuda install
	$(MAKE) install_cuda
else
	echo running x86 install
	$(MAKE) install_x86
endif

# Do everything with cuda.
install_cuda: deps download sudo_pre_install_clean build_cuda sudo_install clean

# CPU install
install_x86: deps download sudo_pre_install_clean build_cuda sudo_install clean

# Install system wide.
sudo_install:
	cd $(TMP_DIR)opencv/opencv-$(OPENCV_VERSION)/build
	$(SUDO) $(MAKE) install
	$(SUDO) ldconfig
	cd -

