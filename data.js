// Description: This file contains the code to load the data from the file system and convert it into tensors.

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const IMAGE_WIDTH = 224;
const IMAGE_HEIGHT = 224;
const BATCH_SIZE = 64;
const SHUFFLE_BUFFER_SIZE = 1024;

const NUM_CLASSES = 5;

const classNames = ['downdog', 'goddess', 'plank', 'tree', 'warrior2'];

function* arrayIterator(arrayOfArrays) {
    for (const array of arrayOfArrays) {
      yield array;
    }
}

async function batchedFilesInDirectory(directoryPath) {
    const result = [];
  
    // Read the directory
    const subdirectories = fs.readdirSync(directoryPath, { withFileTypes: true });
  
    subdirectories.forEach((subdirectory) => {
        if (subdirectory.isDirectory()) {
            const folderName = subdirectory.name;
            const folderPath = path.join(directoryPath, folderName);
    
            // Read files in the subdirectory
            const files = fs.readdirSync(folderPath);
    
            // Create objects and push to the result array
            files.forEach((file) => {
                result.push({ fileName : file, folderName: `${folderName}` });
            });
        }
    });

    const retarr = await tf.data.array(result).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).toArray(); //
    // console.log("return" ,retarr);
    const retval = await convertTensorsToStrings(retarr);
    const iterator = arrayIterator(retval);
    // console.log(await convertTensorsToStrings(retarr));
  
    return iterator;
}

async function convertTensorsToStrings(tensorObjects) {
    const convertedObjects = [];

    for (const obj of tensorObjects) {
        const convertedObj = {
            fileName: await obj.fileName.data(),
            folderName: await obj.folderName.data(),
        };

        convertedObjects.push(convertedObj);
    }

    return convertedObjects.map(obj => obj.fileName.map((fileName, i) => ({
        fileName,
        folderName: obj.folderName[i],
    })));
}

async function loadImagesAndLabels(directory, batchFilesLabelled) {
    const concatenatedImages = [];
    const labels = [];
    
    // #### Issue this needs to be fixed (even though are initally shuffled, enter in an orderly manner throught this)
    classNames.forEach((className, classIndex) => {
        const classDir = `${directory}/${className}`;
        const modifiedObjects = batchFilesLabelled.filter(obj => obj.folderName === className);
        // modifiedObjects.forEach((obj) => {
        //     const fileName = obj.fileName;
        //     const folderName = obj.folderName;
        //     #### console.log(`File Name: ${fileName}, File Path: ${directory}/${folderName}/${fileName}`);
        // });
    
        const resizedImages = modifiedObjects.map((file) => {
            const filePath = `${classDir}/${file.fileName}`;
            try {
                const buffer = fs.readFileSync(filePath);
                const decodedImage = tf.node.decodeImage(buffer);
        
                const alphaCleanedImage = tf.tidy(() => {
                    if (decodedImage.shape[2] === 4) {
                        // If the image has 4 channels (RGBA), remove the alpha channel
                        return decodedImage.slice([0, 0, 0], [-1, -1, 3]);
                    } else if (decodedImage.shape[2] === 1) {
                        return tf.stack([decodedImage, decodedImage, decodedImage], 2);
                    } else {
                        return decodedImage;
                    }
                });
        
                const reshapedImage = tf.div(alphaCleanedImage, 255);
        
                // Check if the ranks match before resizing
                if (reshapedImage.shape.length === 3 && reshapedImage.shape[2] === 3) {
                    const resizedImage = tf.image.resizeBilinear(reshapedImage, [IMAGE_WIDTH, IMAGE_HEIGHT]);
                    return resizedImage;
                } else {
                    console.error(`Invalid image shape: ${reshapedImage.shape}`);
                    return null; // or handle the error in an appropriate way
                }
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

        const classLabels = [...Array.from({ length: validResizedImages.length })].map(x => Array(NUM_CLASSES).fill(0));
        classLabels.forEach((label) => label[classIndex]=1);
        classLabels.forEach((label) => labels.push(tf.tensor1d(label)));
        // const classLabelTensor = tf.tensor2d(classLabels, [files.length, NUM_CLASSES]);
        // labels.push(...classLabelTensor);
    });

    const firstAxis = 0;

    // const imagesTensor = tf.concat(concatenatedImages, firstAxis).reshape([concatenatedImages.length, IMAGE_HEIGHT, IMAGE_WIDTH, 3]);\
    const imagesTensor = concatenatedImages;
    console.log(concatenatedImages.map((image, index) =>image.shape));
    
    // const labelsTensor = tf.concat(labels, firstAxis).reshape([labels.length, NUM_CLASSES]);
    const labelsTensor = labels;
    // labelsTensor.forEach((label) => label.print(verbose=true));
    // labelsTensor.print(verbose=true);

    return [imagesTensor, labelsTensor];
};


class PoseDataset {
    constructor() {
        this.trainDatasetIterator = null; 
        this.testDatasetIterator = null; 
        this.dataset = null;
        this.trainDirectoryPath = './DATASET/TRAIN';
        this.testDirectoryPath = './DATASET/TEST';
        
        this.trainBatchIndex = 0;
        this.testBatchIndex = 0;
        this.trainBatchFiles = null;
        this.testBatchFiles = null;
    }

    async loadData() {
        this.trainDatasetIterator = await batchedFilesInDirectory(this.trainDirectoryPath);
        this.testDatasetIterator = await batchedFilesInDirectory(this.testDirectoryPath);   
    }

    async getTrainData() {
        const result = this.trainDatasetIterator.next();
        if (!result.done) {
            this.trainBatchFiles = result.value;
            this.trainBatchIndex++;
        } else { // results in looping
            this.trainBatchIndex = 1;
            this.trainDatasetIterator = await batchedFilesInDirectory(this.trainDirectoryPath);
            const result = this.trainDatasetIterator.next();
            this.trainBatchFiles = result.value;
        }
        return this.getData_();
    }
    
    // getTestData() {
    //     const result = this.testDatasetIterator.next();
    //     if (!result.done) {
    //         this.testBatchFiles = result.value;
    //         this.testBatchIndex++;
    //         return this.getData_(false);
    //     } else {
    //         this.testBatchIndex = 0;
    //         this.testBatchFiles = null;
    //         return null;
    //     }
    // }

    async getData_() {
        const imagesIndex = 0;
        const labelsIndex = 1;
        // if (isTrainingData) {
        const [trainImages, trainLabels] = await loadImagesAndLabels(this.trainDirectoryPath, this.trainBatchFiles);
        this.dataset = await Promise.all([trainImages, trainLabels]);
        // } else {
        //     const [testImages, testLabels] = await loadImagesAndLabels(this.testDirectoryPath, this.testBatchFiles);
        //     this.dataset = await Promise.all([testImages, testLabels]);
        // }

        // console.log(this.dataset);
        
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

