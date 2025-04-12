import testData from './testData.json' with {type: 'json'}

let nn


function initializeNeuralNetwork() {
    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({task: 'classification', debug: true})
    const options = {
        model: "./model/model.json",
        metadata: "./model/model_meta.json",
        weights: "./model/model.weights.bin",
    }
    nn.load(options, completeModelTraining)
}

const completeModelTraining = async () => {
    let correct = 0;

    console.log(testData.data)
    for (const dataPoint of testData) {
        nn.classify(dataPoint.data, (result) => {
            console.log(`Actual label: ${dataPoint.label}`);
            console.log(`Predicted label: ${result[0].label}`);
            if (result[0].label === dataPoint.label) {
                correct++;
                console.log(correct)
            }
            let accuracy = (correct / testData.length) * 100;
            console.log(`Accuracy: ${accuracy.toFixed(2)}%`);
            console.log(`Got ${correct} correct answers out of ${testData.length}`)


        })
        // let sampleDataPoint = testData[i]
        // neuralNetwork.classify(sampleDataPoint.data, (classificationResult) => {
        //     console.log(`Actual label: ${sampleDataPoint.label}`);
        //     console.log(`Predicted label: ${classificationResult[0].label}`);
        //     if (classificationResult[0].label === sampleDataPoint.label) {
        //         correct++
        //     }
        //     i++
        //     if (i < testData.length) {
        //         completeModelTraining()
        //     } else {
        //         let accuracy = (correct / testData.length) * 100;
        //         console.log(`Accuracy: ${accuracy.toFixed(2)}%`);
        //     }
        // })
    }
}

initializeNeuralNetwork()
