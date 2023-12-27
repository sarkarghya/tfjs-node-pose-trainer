// Desc: Create and train a model for classifying images of poses.

const tf = require('@tensorflow/tfjs-node');


const BATCH_SIZE = 16;
const NUM_EPOCHS = 50;

const IMAGE_WIDTH = 224;
const IMAGE_HEIGHT = 224;
const NUM_CLASSES = 5;

const createModel = async () => {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, 3],
        kernelSize: [3,3],
        padding: 'same', 
        filters: 32,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2})); // 2x2 pool size, 2x2 stride
    model.add(tf.layers.dropout({rate: 0.25}));

    model.add(tf.layers.conv2d({
        kernelSize: [3,3],
        padding: 'same', 
        filters: 64, 
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tf.layers.dropout({rate: 0.25}));

    model.add(tf.layers.conv2d({
        kernelSize: [3,3],
        padding: 'same', 
        filters: 128, 
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tf.layers.dropout({rate: 0.25}));

    model.add(tf.layers.flatten({}));

    model.add(tf.layers.dense({units: 512, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.25}));
    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'
    }));
  
    return model;
};

const trainModel = async (model, images, labels) => {

    const learningRate = 0.001;
    const optimizer = tf.train.adam(learningRate);
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    model.fit(images, labels, {
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS,
        shuffle: true
    });

};

module.exports = {
    createModel,
    trainModel  
};