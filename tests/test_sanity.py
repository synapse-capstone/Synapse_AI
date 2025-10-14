import importlib, sys

def test_imports():
    pkgs = [
        "openai",
        "spacy",
        "google.cloud.texttospeech",
        "sounddevice",
        "soundfile",
    ]
    # Python 3.13에서는 pydub가 audioop 대체 모듈을 찾지 못함
    if sys.version_info < (3, 13):
        pkgs.append("pydub")

    for p in pkgs:
        assert importlib.import_module(p)
