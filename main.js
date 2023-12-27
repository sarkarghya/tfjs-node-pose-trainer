// Description: Main file for the project. This file will be used to load the data and train the model.

const data = require('./data');
const model = require('./model');

const main = async () => {
    await data.loadData();
    // const trainData = data.getTrainData();
    // console.log(trainData);

    const {images: trainImages, labels: trainLabels} = data.getTrainData();
    console.log(trainImages, trainLabels);
    // const {imagesTensor, labelsTensor} = await data.loadData();  
    // console.log(imagesTensor, labelsTensor);
    const cnnModel = await model.createModel();
    cnnModel.summary();
    // await model.trainModel(cnnModel, trainData);
    await model.trainModel(cnnModel, trainImages, trainLabels);
};

main();





  
