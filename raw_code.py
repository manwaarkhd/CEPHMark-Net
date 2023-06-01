"""
def fusion_block(inputs: tf.keras.layers, filters: int, block_name: str):
    x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), name="fusion_" + block_name + "_conv2d")(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.75, epsilon=1e-5, name="fusion_" + block_name + "_batchnorm")(x)
    x = tf.keras.layers.Activation(activation="relu", name="fusion_" + block_name + "_activation")(x)

    return x


def atrous_spatial_pyramid_pooling(
    inputs: tf.keras.layers,
    feature_map_depth: int,
    rates: list = [0, 1, 2, 4]
):
    def dilation_block(
        inputs: tf.keras.layers,
        filters: int,
        padding: int,
        name: str,
        dilation_rate: int,
    ):
        block_name = "dilation_" + name

        x = tf.keras.layers.ZeroPadding2D(padding, name=block_name + "_padding")(inputs)
        if dilation_rate == 0:
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), name=block_name + "_conv2d")(x)
        else:
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), dilation_rate=dilation_rate, name=block_name + "_conv2d")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.85, epsilon=1e-5, name=block_name + "_batchnorm")(x)
        x = tf.keras.layers.Activation(activation="relu", name=block_name + "_activtion")(x)

        return x

    filters = feature_map_depth // 4

    block1 = dilation_block(inputs, filters, padding=rates[0], dilation_rate=rates[0], name="block1")
    block2 = dilation_block(inputs, filters, padding=rates[1], dilation_rate=rates[1], name="block2")
    block3 = dilation_block(inputs, filters, padding=rates[2], dilation_rate=rates[2], name="block3")
    block4 = dilation_block(inputs, filters, padding=rates[3], dilation_rate=rates[3], name="block4")

    return tf.keras.layers.Concatenate(name="dilation_module_concat")([block1, block2, block3, block4])


class ROIPool2D(tf.keras.layers.Layer):

    def __init__(self, crop_size: tuple, offset: float = 0.03215):
        super(ROIPool2D, self).__init__()
        self.crop_size = crop_size
        self.offset = offset

    def call(self, inputs, *args, **kwargs):
        feature_map, roi_proposals = inputs

        if feature_map.shape[1] and feature_map.shape[2] is not None:
            x1, y1, x2, y2 = tf.split(roi_proposals, 4, axis=-1)
            roi_proposals = tf.concat([y1, x1, y2 + self.offset, x2 + self.offset], axis=-1)

        roi_proposals = tf.stop_gradient(roi_proposals)
        cropped_maps = tf.image.crop_and_resize(
            feature_map,
            roi_proposals,
            box_indices=tf.range(tf.shape(roi_proposals)[0]),
            crop_size=self.crop_size,
            method="nearest"
        )

        return cropped_maps


class ROIAlign2D(tf.keras.layers.Layer):

    def __init__(self, crop_size: tuple, name: str):
        super(ROIAlign2D, self).__init__()
        self.crop_size = crop_size
        self._name = name

    def call(self, inputs, *args, **kwargs):
        feature_maps, roi_proposals = inputs

        if roi_proposals.shape[0] is not None:
            x1, y1, x2, y2 = tf.split(roi_proposals, 4, axis=-1)
            roi_proposals = tf.concat([y1, x1, y2, x2], axis=-1)

        roi_proposals = tf.stop_gradient(roi_proposals)

        cropped_maps = []
        for index in range(len(feature_maps)):
            cropped_featuremap = tf.image.crop_and_resize(
                feature_maps[index],
                roi_proposals[index],
                box_indices=tf.range(tf.shape(roi_proposals)[1]),
                crop_size=self.crop_size,
                method="bilinear"
            )
            cropped_maps.append(cropped_featuremap)
        cropped_maps = tf.concat(cropped_maps, axis=-1)

        return cropped_maps


isbiplus_dataset_root = Paths.dataset_root_path("isbi+")
isbi_dataset_root = Paths.dataset_root_path("isbi")

train_data = Dataset(dataset_root_path=isbiplus_dataset_root, name="isbi+", mode="train", batch_size=4, shuffle=True)
valid_data = Dataset(dataset_root_path=isbi_dataset_root, name="isbi", mode="valid", shuffle=False)
test_data  = Dataset(dataset_root_path=isbi_dataset_root, name="isbi", mode="test", shuffle=False)

input_1 = tf.keras.layers.Input(shape=(800, 640, 3), name="cephalogram")
input_2 = tf.keras.layers.Input(shape=(None, 4), name="proposal")

backbone = tf.keras.applications.vgg19.VGG19(
    include_top=False,
    input_tensor=input_1,
    weights=None
)
block4_output = backbone.get_layer("block4_pool").output
block5_output = backbone.get_layer("block5_pool").output
base_layer = backbone.output

# block3_output = backbone.get_layer("block3_pool").output
# block4_output = backbone.get_layer("block4_pool").output
# block5_output = backbone.get_layer("block5_pool").output
#
# block3_output = fusion_block(block3_output, filters=256, block_name="block3")
# block4_output = fusion_block(block4_output, filters=512, block_name="block4")
# block5_output = fusion_block(block5_output, filters=512, block_name="block5")
#
# feat_height, feat_width = block5_output.shape[1:3]
# block3_output = tf.keras.layers.Resizing(height=feat_height, width=feat_width, interpolation="bilinear", name="fusion_block3_downsample")(block3_output)
# block4_output = tf.keras.layers.Resizing(height=feat_height, width=feat_width, interpolation="bilinear", name="fusion_block4_downsample")(block4_output)
# block5_output = tf.keras.layers.Concatenate(name="fusion_block_concat")([block3_output, block4_output, block5_output])
#
# base_layers = atrous_spatial_pyramid_pooling(inputs=block5_output, feature_map_depth=1024, rates=[0, 1, 2, 4])

x = tf.keras.layers.Flatten(name="detection_block_flatten")(base_layer)
output_1 = tf.keras.layers.Dense(units=38, activation="linear", use_bias=False, name="detection_block_output")(x)

landmark_detection_network = tf.keras.models.Model(
    inputs=input_1,
    outputs=output_1,
    name="landmark_detection_network"
)
landmark_detection_network.load_weights("./logs/weights/VGG19/model/")
landmark_detection_network.trainable = False

x = ROIAlign2D(crop_size=(5, 5), name="refinement_block_roialign")([[block4_output, block5_output], input_2])
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="same", name="refinement_block_conv2d")(x)
x = tf.keras.layers.Flatten(name="refinement_block_flatten")(x)
output_2 = tf.keras.layers.Dense(units=2, activation="linear", use_bias=False, name="refinement_block_output")(x)

landmark_refinement_network = tf.keras.models.Model(
    inputs=[input_1, input_2],
    outputs=output_2,
    name="landmark_refinement_network"
)

model = tf.keras.models.Model(
    inputs=[input_1, input_2],
    outputs=[output_1, output_2],
    name="network"
)
tf.keras.utils.plot_model(model, show_shapes=True)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
loss_fn = MeanSquaredError()
metric_fn = MeanRadialError()
feat_stride = 32
epochs = 30

for epoch in range(1, epochs + 1):
    num_digits = len(str(epochs))
    fmt = "{:" + str(num_digits) + "d}"
    print("\nEpoch: " + fmt.format(epoch) + "/" + fmt.format(epochs) + "")

    losses = []
    metrics = [[], [], []]
    for index in range(len(train_data)):
        images, landmarks = train_data[index]
        images = rescale_input(images, scale=(1/255), offset=0)
        image_height, image_width = images.shape[1:-1]
        feat_height, feat_width = image_height/feat_stride, image_width/feat_stride

        with tf.GradientTape() as tape:
            true_landmarks = encode_cephalometric_landmarks(landmarks, image_height, image_width)
            pred_landmarks = landmark_detection_network(inputs=images, training=False)
            pred_landmarks = tf.stack([pred_landmarks[:, 0::2], pred_landmarks[:, 1::2]], axis=-1)

            clp_true_landmarks = decode_cephalometric_landmarks(true_landmarks, height=2400, width=1935)
            clp_pred_landmarks = decode_cephalometric_landmarks(pred_landmarks, height=2400, width=1935)

            loss_1 = loss_fn(true_landmarks, pred_landmarks)
            metric_1 = metric_fn(clp_true_landmarks, clp_pred_landmarks)

            block4_proposals = craniofacial_landmark_regions(pred_landmarks, height=(image_height/16), width=(image_width/16), size=7)
            block5_proposals = craniofacial_landmark_regions(pred_landmarks, height=(image_height/32), width=(image_width/32), size=5)
            proposals = tf.stack([block4_proposals, block5_proposals])

            landmark_index = 9
            candidate_regions = proposals[:, :, landmark_index, :]
            true_locations = true_landmarks[:, landmark_index, :]
            pred_locations = landmark_refinement_network(inputs=[images, candidate_regions], training=True) + pred_landmarks[:, landmark_index, :]

            clr_true_landmarks = tf.stack([true_locations[:, 0] * 1935, true_locations[:, 1] * 2400], axis=-1)
            clr_pred_landmarks = tf.stack([pred_locations[:, 0] * 1935, pred_locations[:, 1] * 2400], axis=-1)

            loss_2 = loss_fn(true_locations, pred_locations)

            metric_2 = tf.reduce_mean(
                tf.sqrt(
                    tf.add(
                        tf.square(clp_true_landmarks[:, :, 0] - clp_pred_landmarks[:, :, 0]),
                        tf.square(clp_true_landmarks[:, :, 1] - clp_pred_landmarks[:, :, 1])
                    )
                ),
                axis=0
            )[landmark_index]

            metric_3 = tf.reduce_mean(
                tf.sqrt(
                    tf.add(
                        tf.square(clr_true_landmarks[:, 0] - clr_pred_landmarks[:, 0]),
                        tf.square(clr_true_landmarks[:, 1] - clr_pred_landmarks[:, 1])
                    )
                )
            )

            loss = tf.stack([tf.reshape(loss_1, shape=(1,)), tf.reshape(loss_2, shape=(1,))], axis=0)

        gradients = tape.gradient(loss_2, landmark_refinement_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, landmark_refinement_network.trainable_variables))

        losses.append(tf.reduce_sum(loss))
        metrics[0].append(metric_1)
        metrics[1].append(metric_2)
        metrics[2].append(metric_3)

        print("\rloss: {:.3f} - mean_radial_error: {:.3f} - landmark_detection_error: {:.3f} - landmark_refinment_error: {:.3f}".format(tf.reduce_mean(losses), tf.reduce_mean(metrics[0]), tf.reduce_mean(metrics[1]), tf.reduce_mean(metrics[2])), end="")

    print("\ntrain_loss: {:.3f} - mean_radial_error: {:.3f} - landmark_detection_error: {:.3f} - landmark_refinment_error: {:.3f}".format(tf.reduce_mean(losses), tf.reduce_mean(metrics[0]), tf.reduce_mean(metrics[1]), tf.reduce_mean(metrics[2])), end="")

    losses = []
    metrics = [[], [], []]
    for index in range(len(valid_data)):
        images, landmarks = valid_data[index]
        images = rescale_input(images, scale=(1 / 255), offset=0)
        image_height, image_width = images.shape[1:-1]
        feat_height, feat_width = image_height / feat_stride, image_width / feat_stride

        true_landmarks = encode_cephalometric_landmarks(landmarks, image_height, image_width)
        pred_landmarks = landmark_detection_network(inputs=images, training=False)
        pred_landmarks = tf.stack([pred_landmarks[:, 0::2], pred_landmarks[:, 1::2]], axis=-1)

        clp_true_landmarks = decode_cephalometric_landmarks(true_landmarks, height=2400, width=1935)
        clp_pred_landmarks = decode_cephalometric_landmarks(pred_landmarks, height=2400, width=1935)

        loss_1 = loss_fn(true_landmarks, pred_landmarks)
        metric_1 = metric_fn(clp_true_landmarks, clp_pred_landmarks)

        block4_proposals = craniofacial_landmark_regions(pred_landmarks, height=(image_height/16), width=(image_width/16), size=5)
        block5_proposals = craniofacial_landmark_regions(pred_landmarks, height=(image_height/32), width=(image_width/32), size=3)
        proposals = tf.stack([block4_proposals, block5_proposals])

        landmark_index = 9
        candidate_regions = proposals[:, :, landmark_index, :]
        true_locations = true_landmarks[:, landmark_index, :]
        pred_locations = landmark_refinement_network(inputs=[images, candidate_regions], training=False) + pred_landmarks[:, landmark_index, :]

        clr_true_landmarks = tf.stack([true_locations[:, 0] * 1935, true_locations[:, 1] * 2400], axis=-1)
        clr_pred_landmarks = tf.stack([pred_locations[:, 0] * 1935, pred_locations[:, 1] * 2400], axis=-1)

        # landmark_index = 9
        # candidate_landmarks = true_landmarks[:, landmark_index, :]
        # candidate_regions = proposals[:, landmark_index, :]
        #
        # true_locations = encode_landmark_locations(candidate_regions, candidate_landmarks, feat_height, feat_width, size=crop_size)
        # pred_locations = landmark_refinement_network(inputs=[images, candidate_regions])
        #
        # clr_true_landmarks = tf.stack([true_locations[:, 0] * 1935, true_locations[:, 1] * 2400], axis=-1)
        # clr_pred_landmarks = tf.stack([pred_locations[:, 0] * 1935, pred_locations[:, 1] * 2400], axis=-1)

        metric_2 = tf.reduce_mean(
            tf.sqrt(
                tf.add(
                    tf.square(clp_true_landmarks[:, :, 0] - clp_pred_landmarks[:, :, 0]),
                    tf.square(clp_true_landmarks[:, :, 1] - clp_pred_landmarks[:, :, 1])
                )
            ),
            axis=0
        )[landmark_index]

        metric_3 = tf.reduce_mean(
            tf.sqrt(
                tf.add(
                    tf.square(clr_true_landmarks[:, 0] - clr_pred_landmarks[:, 0]),
                    tf.square(clr_true_landmarks[:, 1] - clr_pred_landmarks[:, 1])
                )
            )
        )

        loss = tf.stack([tf.reshape(loss_1, shape=(1,)), tf.reshape(loss_2, shape=(1,))], axis=0)

        losses.append(tf.reduce_sum(loss))
        metrics[0].append(metric_1)
        metrics[1].append(metric_2)
        metrics[2].append(metric_3)

    print("\nvalid_loss: {:.3f} - mean_radial_error: {:.3f} landmark_detection_error: {:.3f} - landmark_refinment_error: {:.3f}".format(tf.reduce_mean(losses), tf.reduce_mean(metrics[0]), tf.reduce_mean(metrics[1]), tf.reduce_mean(metrics[2])), end="")

    weights_root_path = os.path.join(".", "logs", "weights", "VGG16", "model", "")
    # model.save_weights(weights_root_path)
"""

"""
epoch = 1
weights_root_path = os.path.join(".", "logs", "weights", "VGG16", "ISBI150", str(epoch), "")
model.load_weights(weights_root_path)
losses = []
metrics = []
for index in range(len(test_data)):
    images, landmarks = test_data[index]
    images = rescale_input(images, scale=(1 / 255), offset=0)
    batch_size, image_height, image_width = images.shape[:-1]
    true_landmarks = encode_cephalometric_landmarks(landmarks, image_height, image_width)
    pred_landmarks = model(inputs=images, training=False)
    pred_landmarks = tf.stack([pred_landmarks[:, 0::2], pred_landmarks[:, 1::2]], axis=-1)
    clp_true_landmarks = decode_cephalometric_landmarks(true_landmarks, height=2400, width=1935)
    clp_pred_landmarks = decode_cephalometric_landmarks(pred_landmarks, height=2400, width=1935)
    loss = loss_fn(true_landmarks, pred_landmarks)
    metric = metric_fn(clp_true_landmarks, clp_pred_landmarks)
    losses.append(loss)
    metrics.append(metric)
print("\ntest_loss: {:.3f} - mean radial error: {:.3f}".format(tf.reduce_mean(losses), tf.reduce_mean(metrics)), end="")
"""

"""
index = 10
image, landmarks = valid_data[index]
image_height, image_width = image.shape[1:3]
heatmap = get_gradcam_heatmap(
    inputs=[image],
    model=model,
    last_layer_name="block5_conv3",
    height=image_height,
    width=image_width,
    feat_dims=38
)
# fig = plt.figure(figsize=(19.20,10.80))
plt.imshow(image[0])
plt.imshow(heatmap, cmap="jet", alpha=0.75)
plt.xticks([])
plt.yticks([])
# plt.savefig("grad-cam-image-193.svg", format="eps", dpi=1080)
plt.show()
"""

# network = Network(backbone_name="vgg16", freeze_backbone=False)
#
# history = train(
#     dataset=train_data,
#     network=network,
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#     losses=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_squared_error],
#     metrics=[MeanIoU(), MeanChebyshevDistance()],
#     epochs=3,
#     valid_data=valid_data
# )

"""
index = 0
images, landmarks = valid_data[index]
image_height, image_width = images.shape[1:3]

true_bboxes = craniofacial_region_proposals(landmarks, image_height, image_width)
pred_bboxes = network.cfd(inputs=images, training=False)
pred_bboxes = clip_bounding_boxes(pred_bboxes)

image = np.copy(tf.squeeze(images))

bbox = decode_bounding_boxes(true_bboxes, image_height, image_width)
bbox = transform_bounding_boxes(bbox, mode="xyxy")
cv2.rectangle(image, (int(bbox[:, :, 0]), int(bbox[:, :, 1])), (int(bbox[:, :, 2]), int(bbox[:, :, 3])), color=(0, 255, 0), thickness=3)

bbox = decode_bounding_boxes(pred_bboxes, image_height, image_width)
bbox = transform_bounding_boxes(bbox, mode="xyxy")
cv2.rectangle(image, (int(bbox[:, :, 0]), int(bbox[:, :, 1])), (int(bbox[:, :, 2]), int(bbox[:, :, 3])), color=(255, 0, 0), thickness=3)

plt.imshow(image)
plt.show()
"""

"""
index = 0
images, landmarks = valid_data[index]
image_height, image_width = images.shape[1:3]

true_bboxes = craniofacial_region_proposals(landmarks, image_height, image_width)
pred_bboxes = network.cfd(inputs=images, training=False)
clip_bboxes = clip_bounding_boxes(pred_bboxes)

true_regions = encode_cephalometric_landmarks(landmarks, image_height, image_width)
pred_regions = network.clp(inputs=[images, tf.reshape(pred_bboxes, (-1, 4))], training=True)
pred_regions = tf.reshape(pred_regions, shape=(-1, cfg.NUM_LANDMARKS*2))
pred_regions = tf.stack([pred_regions[:, 0::2], pred_regions[:, 1::2]], axis=-1)

image = np.copy(tf.squeeze(images))

plt.imshow(image)
landmarks = decode_cephalometric_landmarks(true_regions, image_height, image_width)
plt.scatter(landmarks[:, :, 0], landmarks[:, :, 1], color="green", s=[5]*cfg.NUM_LANDMARKS)
landmarks = decode_cephalometric_landmarks(pred_regions, image_height, image_width)
plt.scatter(landmarks[:, :, 0], landmarks[:, :, 1], color="red", s=[5]*cfg.NUM_LANDMARKS)
plt.show()
"""

"""
index = 10
image, landmarks = valid_data[index]
image_height, image_width = image.shape[1:3]

heatmap = get_gradcam_heatmap(
    inputs=[image],
    model=network.cfd,
    last_layer_name="block5_conv3",
    height=image_height,
    width=image_width,
    feat_indices=4
)

# fig = plt.figure(figsize=(19.20,10.80))
plt.imshow(image[0])
plt.imshow(heatmap, cmap="jet", alpha=0.75)
plt.xticks([])
plt.yticks([])
# plt.savefig("grad-cam-image-193.svg", format="eps", dpi=1080)
plt.show()
"""

""""
index = 10
image, landmarks = valid_data[index]
image_height, image_width = image.shape[1:3]

pred_bboxes = network.cfd(inputs=image, training=False)
pred_bboxes = clip_bounding_boxes(pred_bboxes)

bbox = decode_bounding_boxes(pred_bboxes, image_height, image_width)
bbox = transform_bounding_boxes(bbox, mode="xyxy")
bbox = tf.cast(tf.squeeze(bbox), dtype=tf.int32)
x1, y1, x2, y2 = bbox.numpy()

heatmap = get_gradcam_heatmap(
    inputs=[image, tf.reshape(pred_bboxes, shape=(-1, 4))],
    model=network.clp,
    last_layer_name="clp_block_conv2d",
    height=(y2 - y1),
    width=(x2 - x1),
    feat_indices=38
)

img = np.copy(image[0])
img = img[y1:y2, x1:x2]
# fig = plt.figure(figsize=(19.20,10.80))

plt.imshow(img)
plt.imshow(heatmap, cmap="jet", alpha=0.75)
plt.xticks([])
plt.yticks([])
# plt.savefig("grad-cam-image-193.svg", format="eps", dpi=1080)
plt.show()
"""

"""
tf.keras.utils.plot_model(network.model, show_shapes=True)
"""