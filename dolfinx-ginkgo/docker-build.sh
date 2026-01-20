#!/bin/bash
# Build and test dolfinx-ginkgo in Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="dolfinx-ginkgo"
IMAGE_TAG="latest"

# Build the Docker image
echo "=== Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG} ==="
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${SCRIPT_DIR}"

# Build dolfinx-ginkgo library inside container
echo ""
echo "=== Building dolfinx-ginkgo library ==="
docker run --rm -v "${SCRIPT_DIR}:/home/fenics/dolfinx-ginkgo" \
    -w /home/fenics/dolfinx-ginkgo \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash -c "
        mkdir -p build && cd build && \
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DDOLFINX_GINKGO_BUILD_TESTS=ON \
            -DDOLFINX_GINKGO_BUILD_EXAMPLES=ON \
            -DDOLFINX_GINKGO_BUILD_PYTHON=OFF && \
        make -j\$(nproc)
    "

# Run tests
echo ""
echo "=== Running tests ==="
docker run --rm -v "${SCRIPT_DIR}:/home/fenics/dolfinx-ginkgo" \
    -w /home/fenics/dolfinx-ginkgo/build \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash -c "
        echo '--- Running MPI integration test ---' && \
        mpirun --allow-run-as-root -n 2 ./tests/test_distributed_matrix
    "

echo ""
echo "=== Build and tests complete ==="
