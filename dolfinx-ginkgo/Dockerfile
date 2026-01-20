# Dockerfile for dolfinx-ginkgo
# Builds DOLFINx + Ginkgo 1.11.0 for GPU-accelerated solvers

FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

LABEL maintainer="CardioEMI Project"
LABEL description="DOLFINx with Ginkgo 1.11.0 for GPU-accelerated linear solvers"

# Install Ginkgo build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Ginkgo 1.11.0 from source
ARG GINKGO_VERSION=v1.11.0
ARG BUILD_JOBS=2
RUN git clone --depth 1 --branch ${GINKGO_VERSION} \
        https://github.com/ginkgo-project/ginkgo.git /tmp/ginkgo && \
    cd /tmp/ginkgo && mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DGINKGO_BUILD_TESTS=OFF \
        -DGINKGO_BUILD_EXAMPLES=OFF \
        -DGINKGO_BUILD_BENCHMARKS=OFF \
        -DGINKGO_BUILD_REFERENCE=ON \
        -DGINKGO_BUILD_OMP=ON \
        -DGINKGO_BUILD_MPI=ON \
        -DGINKGO_BUILD_CUDA=OFF \
        -DGINKGO_BUILD_HIP=OFF \
        -DGINKGO_BUILD_SYCL=OFF \
    && make -j${BUILD_JOBS} \
    && make install \
    && rm -rf /tmp/ginkgo

# Update library cache
RUN ldconfig

# Install nanobind for Python bindings
RUN pip install --no-cache-dir nanobind

# Set working directory
WORKDIR /home/fenics

# Default command
CMD ["/bin/bash"]
