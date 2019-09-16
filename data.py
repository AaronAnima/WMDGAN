import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags

import ipdb


# 1 5_o_Clock_Shadow：刚长出的双颊胡须
# 2 Arched_Eyebrows：柳叶眉
# 3 Attractive：吸引人的
# 4 Bags_Under_Eyes：眼袋
# 5 Bald：秃头
# 6 Bangs：刘海
# 7 Big_Lips：大嘴唇
# 8 Big_Nose：大鼻子
# 9 Black_Hair：黑发
# 10 Blond_Hair：金发
# 11 Blurry：模糊的
# 12 Brown_Hair：棕发
# 13 Bushy_Eyebrows：浓眉
# 14 Chubby：圆胖的
# 15 Double_Chin：双下巴
# 16 Eyeglasses：眼镜
# 17 Goatee：山羊胡子
# 18 Gray_Hair：灰发或白发
# 19 Heavy_Makeup：浓妆
# 20 High_Cheekbones：高颧骨
# 21 Male：男性
# 22 Mouth_Slightly_Open：微微张开嘴巴
# 23 Mustache：胡子，髭
# 24 Narrow_Eyes：细长的眼睛
# 25 No_Beard：无胡子
# 26 Oval_Face：椭圆形的脸
# 27 Pale_Skin：苍白的皮肤
# 28 Pointy_Nose：尖鼻子
# 29 Receding_Hairline：发际线后移
# 30 Rosy_Cheeks：红润的双颊
# 31 Sideburns：连鬓胡子
# 32 Smiling：微笑
# 33 Straight_Hair：直发
# 34 Wavy_Hair：卷发
# 35 Wearing_Earrings：戴着耳环
# 36 Wearing_Hat：戴着帽子
# 37 Wearing_Lipstick：涂了唇膏
# 38 Wearing_Necklace：戴着项链
# 39 Wearing_Necktie：戴着领带
# 40 Young：年轻人

#
def get_celebA_attr(Attr_type):
    image_path = "/home/dgx/WMD/data/celebA/celebA"
    CelebA_Attr_file = "/home/dgx/WMD/data/list_attr_celeba.txt"
    labels = []
    with open(CelebA_Attr_file, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Attr_info = Attr_info[2:]
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(image_path, filename)
            if os.path.isfile(filepath_old):
                labels.append(info[Attr_type])
            else:
                print("%d: not found %s\n" % (index, filepath_old))
                not_found_txt.write(line)
    return labels


def get_dataset_train():
    # images = tl.files.load_celebA_dataset('/home/asus/data/celebA/')
    # ipdb.set_trace()
    images = tl.files.load_celebA_dataset(path='../data/celebA')
    # images = images / 127.5 - 1
    labels = get_celebA_attr(21)

    assert len(images) == len(labels)

    def generator_train():
        for image, label in zip(images, labels):
            # ipdb.set_trace()

            yield image.encode('utf-8'), label

    def _map_fn(image_path, label):
        image = tf.io.read_file(image_path)
        # ipdb.set_trace()
        #
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        x = tf.image.convert_image_dtype(image, dtype=tf.float32)


        # M_rotate = tl.prepro.affine_rotation_matrix(angle=(-16, 16))
        # M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5)
        # M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.2))


        # h, w, _ = x.shape
        # M_combined = M_zoom.dot(M_flip).dot(M_rotate)
        # transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
        # x = tl.prepro.affine_transform_cv2(x, transform_matrix, border_mode='replicate')
        #
        # x = tl.prepro.flip_axis(x, axis=1, is_random=True)
        # x = tl.prepro.rotation(x, rg=16, is_random=True, fill_mode='nearest')
        #
        x = tf.image.resize(x, (flags.img_size_h, flags.img_size_w))
        # x = tl.prepro.crop(x, wrg=256, hrg=256, is_random=True)
        # x = x / 127.5 - 1.
        x = x * 2 - 1
        x = tf.image.random_flip_left_right(x)
        return x, label

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.string, tf.int32))
    train_ds = train_ds.shuffle(buffer_size=4096)
    # ds = train_ds.shuffle(buffer_size=4096)
    ds = train_ds.repeat(flags.n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(flags.batch_size_train)
    ds = ds.prefetch(buffer_size=4)  # For concurrency
    return ds, len(images)




