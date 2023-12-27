// Description: This file contains the code to load the data from the file system and convert it into tensors.

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const IMAGE_WIDTH = 224;
const IMAGE_HEIGHT = 224;

const NUM_CLASSES = 5;

const testDir = './DATASET/TEST'; // Replace with "./DATASET/TEST" for real run
const trainDir = './DATASET/TRAIN'; // Replace with "./DATASET/TRAIN" for real run

const classNames = ['downdog', 'goddess', 'plank', 'tree', 'warrior2'];

async function loadImagesAndLabels(directory) {
    const concatenatedImages = [];
    const labels = [];

    classNames.forEach((className, classIndex) => {
        const classDir = `${directory}/${className}`;
        const files = fs.readdirSync(classDir);
    
        const resizedImages = files.map((file) => {
            const filePath = `${classDir}/${file}`;
            try {
                const buffer = fs.readFileSync(filePath);
                const decodedImage = tf.node.decodeImage(buffer);
            
                // const reshapedImage = tf.tidy(() => {
                //     if (decodedImage.shape[2] === 4) {
                //         // If the image has 4 channels (RGBA), remove the alpha channel
                //         return decodedImage.slice([0, 0, 0], [-1, -1, 3]);
                //     } else {
                //         return decodedImage;
                //     }
                // });
                const reshapedImage = tf.div(decodedImage, 255);
                const resizedImage = tf.image.resizeBilinear(reshapedImage, [IMAGE_WIDTH, IMAGE_HEIGHT]); 

                // const resizedImageArray = resizedImageTensor.data();

                // console.log(resizedImage);
                return resizedImage;
            } catch (error) {
                console.error(`Error processing image: ${filePath}`, error);
                return null; // or handle the error in an appropriate way
            }
        });
        const validResizedImages = resizedImages.filter(image => image !== null);
        concatenatedImages.push(...validResizedImages); 

        // const concatenatedClassImages = tf.concat(resizedImages); 
        // console.log(concatenatedClassImages)
        // concatenatedImages.push(concatenatedClassImages);

        const classLabels = [...Array.from({ length: files.length })].map(x => Array(NUM_CLASSES).fill(0));
        classLabels.forEach((label) => label[classIndex]=1);
        classLabels.forEach((label) => labels.push(tf.tensor1d(label)));
        // const classLabelTensor = tf.tensor2d(classLabels, [files.length, NUM_CLASSES]);
        // labels.push(...classLabelTensor);
    });

    const firstAxis = 0;

    // const imagesTensor = tf.concat(concatenatedImages, firstAxis).reshape([concatenatedImages.length, IMAGE_HEIGHT, IMAGE_WIDTH, 3]);\
    const imagesTensor = concatenatedImages;
    console.log(concatenatedImages.map((image, index) =>image.shape));
    // console.log(imagesTensor);
    
    // const labelsTensor = tf.concat(labels, firstAxis).reshape([labels.length, NUM_CLASSES]);
    const labelsTensor = labels;
    // labelsTensor.forEach((label) => label.print(verbose=true));
    // labelsTensor.print(verbose=true);

    return [imagesTensor, labelsTensor];
};

class PoseDataset {
    constructor() {
        this.dataset = null;
        this.trainSize = 0;
        this.testSize = 0;
        this.trainBatchIndex = 0;
        this.testBatchIndex = 0;
    }

    async loadData() {
        const [trainImages, trainLabels] = await loadImagesAndLabels(trainDir);
        const [testImages, testLabels] = await loadImagesAndLabels(testDir);

        this.dataset = await Promise.all([trainImages, trainLabels, testImages, testLabels]);
        this.trainSize = this.dataset[0].length;
        this.testSize = this.dataset[2].length;
    }

    getTrainData() {
        return this.getData_(true);
      }
    
    getTestData() {
        return this.getData_(false);
    }

    getData_(isTrainingData) {
        let imagesIndex;
        let labelsIndex;
        if (isTrainingData) {
            imagesIndex = 0;
            labelsIndex = 1;
        } else {
            imagesIndex = 2;
            labelsIndex = 3;
        }
        const size = this.dataset[imagesIndex].length;
        tf.util.assert(
            this.dataset[labelsIndex].length === size,
            `Mismatch in the number of images (${size}) and ` +
                `the number of labels (${this.dataset[labelsIndex].length})`);
    
        const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 3];

        const firstAxis = 0;
        return {
            images: tf.concat(this.dataset[imagesIndex], firstAxis).reshape(imagesShape),
            labels: tf.concat(this.dataset[labelsIndex], firstAxis).reshape([size, NUM_CLASSES])
        };
    }

}


module.exports = new PoseDataset();

