import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from myModel import check_faces_similarity


def similarImg():
    assert check_faces_similarity("face1.jpg", "face1.jpg")== 0.0

def noImage():
    assert check_faces_similarity("asdasdas", "face1.jpg") == "Один из файлов не существует"