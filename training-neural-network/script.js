import posedata from './posedata.json' with {type: 'json'}

let nn

function startTraining() {
    ml5.setBackend("webgl");
    nn = ml5.neuralNetwork({task: 'classification', debug: true})
    console.log(nn)

    for (const pose of posedata) {
        nn.addData(pose.data, {label: pose.label})
    }
    nn.normalizeData()
    nn.train({epochs: 100}, finishTraining)
}

const finishTraining = async () => {
    console.log("finished training!")
    const demodata = posedata[1].data
    await nn.classify(demodata, (results) => {
        console.log(results[0].label);
        console.log(`I am ${(results[0].confidence.toFixed(2) * 100)} % sure `);
    });
    nn.save("model", () => console.log("model was saved!"))
}

startTraining()