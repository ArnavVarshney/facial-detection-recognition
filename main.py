import argparse

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

assert insightface.__version__ >= '0.3'

parser = argparse.ArgumentParser(description='insightface app test')
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')
args = parser.parse_args()

model_pack_name = 'buffalo_l'
app = FaceAnalysis(name=model_pack_name)
app.prepare(ctx_id=args.ctx, det_size=(args.det_size, args.det_size))


def test(train_embed):
    for _ in range(1, 8):
        test_img = f'img/test/test{_}.jpg'
        test = cv2.imread(test_img)
        test_faces = app.get(test)
        rimg = app.draw_on(test, test_faces)
        cv2.imwrite(f'img/test/output_test_{_}.jpg', rimg)
        feats = np.array([train_embed, test_faces[0].normed_embedding], dtype=np.float32)
        sims = np.dot(feats, feats.T)
        print(sims[0][1], end="\t")


def main():
    embed = []
    for i in range(1, 6):
        print()
        train_img = f'img/train/train{i}.jpg'
        train = cv2.imread(train_img)
        train_faces = app.get(train)
        rimg = app.draw_on(train, train_faces)
        cv2.imwrite(f'img/train/output_train_{i}.jpg', rimg)
        embed.append(np.array(train_faces[0].normed_embedding, dtype=np.float32))
        compiled = np.mean(embed, axis=0)
        test(compiled)


if __name__ == '__main__':
    main()
