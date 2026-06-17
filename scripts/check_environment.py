from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import datasets
import fastapi
import open_clip
import streamlit
import torch


def format_gb(value_in_bytes: int) -> str:
    """Convierte bytes a gigabytes."""
    return f"{value_in_bytes / (1024**3):.2f} GB"


def get_package_version(package_name: str) -> str:
    """Obtiene la versión instalada de un paquete."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NO INSTALADO"


def test_cuda() -> None:
    """Ejecuta una operación sencilla en la GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch no detecta CUDA dentro del entorno virtual."
        )

    device = torch.device("cuda")

    first_tensor = torch.rand((1024, 1024), device=device)
    second_tensor = torch.rand((1024, 1024), device=device)

    result = first_tensor @ second_tensor
    torch.cuda.synchronize()

    if result.device.type != "cuda":
        raise RuntimeError("La operación no fue ejecutada en la GPU.")


def main() -> None:
    """Muestra información y valida las dependencias principales."""
    print("=" * 60)
    print("VERIFICACIÓN DEL ENTORNO")
    print("=" * 60)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    print(f"CUDA de PyTorch: {torch.version.cuda}")

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)

        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {format_gb(properties.total_memory)}")
    else:
        print("GPU: No detectada")
        print("VRAM: No disponible")

    print("-" * 60)
    print(f"FastAPI: {get_package_version('fastapi')}")
    print(f"Streamlit: {get_package_version('streamlit')}")
    print(f"Datasets: {get_package_version('datasets')}")
    print(f"Qdrant Client: {get_package_version('qdrant-client')}")
    print(f"OpenCLIP: {get_package_version('open-clip-torch')}")
    print(f"OpenCLIP importado: {open_clip is not None}")

    print("-" * 60)
    print("Ejecutando prueba de cálculo en GPU...")

    test_cuda()

    print("Prueba CUDA completada correctamente.")
    print("=" * 60)
    print("ENTORNO LISTO")
    print("=" * 60)



if __name__ == "__main__":
    main()