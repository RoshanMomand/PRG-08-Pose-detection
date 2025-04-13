import {PoseLandmarker, FilesetResolver, DrawingUtils} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const enableWebcamButton = document.getElementById("webcamButton");
const sectionLiveView = document.getElementById("liveView");
const correctPoseDiv = document.getElementById("correct-pose-div");
const logPoseButton = document.getElementById('logPose');
const feedbackText = document.getElementById('feedback-text');
const randomPoseElement = document.getElementById('random-pose'); // Element om de willekeurige pose te tonen
const errorHeader = document.createElement('h1');
const mainContainer = document.querySelector('main');
const canvasCtx = canvasElement.getContext("2d");
const img = document.createElement('img');
const drawingUtils = new DrawingUtils(canvasCtx);

let poseLandmarker = undefined;
let webcamRunning = false;

const videoWidth = "600px"
const videoHeight = "400px"

let nn
let resultHandMark
let pose;

//  Haal data op uit model meta json
async function getUniquePoses() {
    try {
        const response = await fetch("./model/model_meta.json");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const metadata = await response.json();

        // Unique label values ophalen
        const uniquePoses = metadata?.outputs?.label?.uniqueValues || [];
        console.log("Unieke poses:", uniquePoses);

        return uniquePoses;
    } catch (error) {
        console.error("Kon de poses niet ophalen:", error);
        return [];
    }
}

/********************************************************************
 // Toon een willekeurige pose in de HTML
 ********************************************************************/
function displayRandomPose(poses) {

    if (poses.length === 0) {
        randomPoseElement.textContent = "Geen poses beschikbaar.";
        return null;
    }

    // Kies een random index en haal de corresponderende pose op
    const randomIndex = Math.floor(Math.random() * poses.length);
    const randomPose = poses[randomIndex];


    // Toon de willekeurige pose in het HTML-element
    randomPoseElement.textContent = randomPose;

    console.log("Willekeurige pose geselecteerd:", randomPose);
    return randomPose;
}

/********************************************************************
 // Creëer het ML5 Neural Network en laad de unieke poses
 ********************************************************************/
function createNeuralNetwork() {
    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({task: 'classification', debug: true})
    const options = {
        model: "./model/model.json",
        metadata: "./model/model_meta.json",
        weights: "./model/model.weights.bin",
    }

    nn.load(options, async () => {
        console.log("Neural Network loading...");

        // Haal unieke poses op
        const uniquePoses = await getUniquePoses();

        // Selecteer en toon een willekeurige pose in de HTML
        const currentPose = displayRandomPose(uniquePoses);

        console.log('Beschikbare poses:', uniquePoses);
        console.log('Momenteel geselecteerde pose:', currentPose);

        createPoseLandmarker();
    });
}

/********************************************************************
 // Creëer MediaPipe PoseLandmarker
 ********************************************************************/
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 2
    });

    enableWebcamButton.addEventListener("click", enableCam);
    logPoseButton.addEventListener('click', () => {
        setTimeout(predictPose, 200);
    });

    enableWebcamButton.innerText = "Start de Game!";
    console.log("poseLandmarker is ready!");
};

/********************************************************************
 // Activeer webcam stream en begin met voorspellen
 ********************************************************************/
function enableCam(event) {
    console.log("start the webcam");
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }

    webcamRunning = true;
    enableWebcamButton.innerText = "Predicting";
    enableWebcamButton.disabled = true;
    enableWebcamButton.remove();

    const constraints = {
        video: {
            width: {ideal: videoWidth},
            height: {ideal: videoHeight}
        }
    };

    // Start webcam stream
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", async () => {
            canvasElement.style.height = videoHeight;
            canvasElement.style.width = videoWidth;
            video.style.height = videoHeight;
            video.style.width = videoWidth;
            predictWebcam();
        });
    });
}

/********************************************************************
 // Continu detecteer poses met de PoseLandmarker
 ********************************************************************/
async function predictWebcam() {
    poseLandmarker.detectForVideo(video, performance.now(), (result) => drawPose(result));

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

/********************************************************************
 // Teken de gedetecteerde poses of log fouten
 ********************************************************************/
function drawPose(result) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (result.landmarks && result.landmarks.length > 0) {
        for (const landmark of result.landmarks) {
            drawingUtils.drawLandmarks(landmark, {radius: 3.0, color: 'red', lineWidth: 1.5});
            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {color: 'blue', lineWidth: 4});
        }
        errorHeader.remove();
        canvasElement.style.opacity = 1;
        canvasElement.style.filter = "none";
        resultHandMark = result.landmarks[0];
    } else {
        errorHeader.textContent = 'No body markers detected, get in view of the camera';
        document.body.append(errorHeader);
        canvasElement.style.opacity = "0.5";
        resultHandMark = null
    }
}

let chances = 3;

/********************************************************************
 // Voorspel de geselecteerde pose met ML5
 ********************************************************************/
async function predictPose() {
    logPoseButton.disabled = true

    // Controleer of we een geldige pose hebben om te voorspellen
    if (!resultHandMark || resultHandMark.length === 0) {
        feedbackText.textContent = "Geen pose data beschikbaar. Zorg dat je goed zichtbaar bent in de camera.";
        logPoseButton.disabled = false;
        logPoseButton.textContent = 'Predict Pose';
        return; // Stop de functie
    }


    let numbersonly = [];
    pose = resultHandMark;
    for (const point of pose) {
        numbersonly.push(point.x, point.y, point.z);
    }

    nn.classify(numbersonly, async (results) => {
        const label = results[0].label;
        console.log(results[0])
        const confidence = (results[0].confidence * 100).toFixed(2);

        if (randomPoseElement.textContent === label) {
            logPoseButton.disabled = true;
            document.getElementById('correct-pose').innerText = 'Goeie pose gemaakt!!!'
            feedbackText.textContent = `Ik ben ${confidence}% zeker dat dit de pose "${label}" is, Je hebt de juiste pose gedaan !!.`;


            setTimeout(async () => {
                const uniquePoses = await getUniquePoses();
                const filteredPoses = uniquePoses.filter(pose => pose !== randomPoseElement.textContent);
                displayRandomPose(filteredPoses);
                feedbackText.textContent = `Nieuwe pose is geladen.`;
                logPoseButton.disabled = false;
                document.getElementById('correct-pose').innerText = 'Correct Pose will be shown here:'
            }, 5000);
        } else {
            if (randomPoseElement.textContent !== label) {
                chances--;
                feedbackText.textContent = `Probeer opnieuw ! Je hebt nog ${chances} pogingen om te voorspellen daarna wordt het verklapte label weergegeven.`;
                if (chances === 0) {
                    logPoseButton.disabled = true;
                    feedbackText.textContent = `Je hebt geen kansen meer, wacht tot de nieuwe pose wordt geladen :(.`;
                    img.src = `./${randomPoseElement.textContent}.jpg`;
                    document.getElementById('correct-pose').innerText = ''
                    img.alt = 'Verklapte pose';
                    correctPoseDiv.append(img);
                    setTimeout(async () => {
                        img.remove();
                        const uniquePoses = await getUniquePoses();
                        const filteredPoses = uniquePoses.filter(pose => pose !== randomPoseElement.textContent);
                        displayRandomPose(filteredPoses);
                        document.getElementById('correct-pose').innerText = 'Correct Pose will be shown here:'
                        feedbackText.textContent = `Nieuwe pose is geladen.`;
                        logPoseButton.disabled = false; // Re-enable the button after resetting
                        chances = 3; // Reset the chances after a new pose is loaded

                    }, 5000);
                } else {
                    logPoseButton.disabled = false; // Re-enable the button for further attempts if chances remain
                }
            }
        }

        logPoseButton.textContent = 'Predict Pose';

    });
}

/********************************************************************
 // Start het neural network
 ********************************************************************/
createNeuralNetwork();