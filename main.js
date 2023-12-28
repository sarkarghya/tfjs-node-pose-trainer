// Description: Main file for the project. This file will be used to load the data and train the model.

const data = require('./data');
const model = require('./model');

const BATCH_SIZE = 64;
const NUM_EPOCHS = 50;
const VALIDATION_SPLIT = 0.2; 

const main = async () => {
    await data.loadData();
    // TODO: check labelling

    // const {imagesTensor, labelsTensor} = await data.loadData();  
    // console.log(imagesTensor, labelsTensor);
    const cnnModel = await model.createModel();
    cnnModel.summary();

    // await model.trainModel(cnnModel, trainImages, trainLabels);
    for (let epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        // Load training data
        const { images: trainImages, labels: trainLabels } = await data.getTrainData();
        // console.log('Trn Img and Lbls', trainImages, trainLabels);
    
        const numValidationSamples = Math.floor(trainImages.shape[0] * VALIDATION_SPLIT);

        const trainingImages = trainImages.slice([numValidationSamples, 0], [-1, -1]);
        const trainingLabels = trainLabels.slice([numValidationSamples, 0], [-1, -1]);

        const validationImages = trainImages.slice([0, 0], [numValidationSamples, -1]);
        const validationLabels = trainLabels.slice([0, 0], [numValidationSamples, -1]);

        await cnnModel.fit(trainingImages, trainingLabels, {
            batchSize: BATCH_SIZE,
            epochs: 1,
            shuffle: true,
            validationData: [validationImages, validationLabels]
        });
    }
};

main();





  
