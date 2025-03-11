from lib.models.convnext import ConvnextBlock, Convnext3dBlock
from lib.models.simple_block import SimpleBlock, Simple3dBlock
from lib.models.generator import SimpleGenerator, LinkerGenerator

def block_factory(block_name):
    if block_name == "convnext":
        return ConvnextBlock, Convnext3dBlock
    if block_name == "simple":
        return SimpleBlock, Simple3dBlock
    raise ValueError(f"Expected block to be 'simple' or 'convnext', but received block {block_name}")

def generator_factory(gen_name):
    if gen_name == "simple":
        return SimpleGenerator
    if gen_name == "linker":
        return LinkerGenerator