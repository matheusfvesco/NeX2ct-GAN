sudo apt update && sudo apt install ffmpeg -y

# Corrige o índice PyPI antigo da imagem NVIDIA
if grep -q 'pypi.ngc.nvidia.com' /usr/pip.conf 2>/dev/null; then
    sudo sed -i.bak \
      's|pypi\.ngc\.nvidia\.com|pypi.nvidia.com|g' \
      /usr/pip.conf
fi

pip install --user -r requirements.txt

pip install --user -e .
