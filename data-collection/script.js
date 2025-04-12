import {PoseLandmarker, FilesetResolver, DrawingUtils} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const enableWebcamButton = document.getElementById("webcamButton");
const logPoseButton = document.getElementById("logPose");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

let poseLandmarker = undefined;
let webcamRunning = false;


let pose;
let allposes = [];
const videoWidth = "600px"
const videoHeight = "400px"

// ********************************************************************
// if webcam access, load landmarker and enable webcam button
// ********************************************************************
function startApp() {

    // hier ga je machine.learn trainen met local storage data


    // klaar met trainen
    const hasGetUserMedia = () => {
        var _a;
        return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia);
    };
    if (hasGetUserMedia()) {
        createPoseLandmarker();
    } else {
        console.warn("getUserMedia() is not supported by your browser");
    }
}

// ********************************************************************
// create mediapipe
// ********************************************************************
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
    logPoseButton.addEventListener("click", () => {
        startLogging(5)
    });
    enableWebcamButton.innerText = "Start de Game!"
    console.log("poselandmarker is ready!")
};


/********************************************************************
 // Continuously grab image from webcam stream and detect it.
 ********************************************************************/
function enableCam(event) {
    console.log("start the webcam")
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }
    webcamRunning = true;
    enableWebcamButton.innerText = "Predicting";
    enableWebcamButton.disabled = true

    const constraints = {
        video: {
            width: {ideal: videoWidth},
            height: {ideal: videoHeight}
        }
    };

    // Activate the webcam stream.
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

// ********************************************************************
// detect poses!!
// ********************************************************************
async function predictWebcam() {
    poseLandmarker.detectForVideo(video, performance.now(), (result) => drawPose(result));

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// ********************************************************************
// draw the poses or log them in the console
// ********************************************************************
function drawPose(result) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // log de coordinaten
    // teken de coordinaten in het canvas

        for (const landmark of result.landmarks) {
            drawingUtils.drawLandmarks(landmark, {radius: 3});
            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        }
        pose = result.landmarks[0]





}


let intervalEnder = false; // Boolean om te voorkomen dat het interval meerdere keren start

function startLogging(seconds) {


    intervalEnder = true;
    let countdown = seconds;
    let loggedPoseCount = 0;
    let formField = document.getElementById('gympose-label').value.trim()
    if (intervalEnder) {
        const countdownInterval = setInterval(() => {
            console.log(`Tijd over: ${countdown} seconden`);
            countdown--;

            // Als de countdown voorbij is, start loggen
            if (countdown < 0) {
                clearInterval(countdownInterval); // Stop de countdown
                console.log("Countdown voltooid. Start met loggen!");

                // Start een interval om poses te loggen
                const intervalEnder = setInterval(() => {
                    logPose(formField); // Log de huidige pose
                    loggedPoseCount++;

                    // Stop logging na een bepaalde hoeveelheid poses (bijv. 125)
                    if (loggedPoseCount >= 100) {
                        clearInterval(intervalEnder); // Stop pose-log interval

                        console.log(" 200 objecten bereiktLogging voltooid - Alle poses zijn opgeslagen.");
                    }
                }, 100);
                intervalEnder = false; // Reset logging status
            }
        }, 1000);
    }


    // Start de interval voor countdown

}


function logPose(formField) {

    let tempPoses = [];
    for (const p of pose) {
        console.log(`Your pose is ${formField} and the data is x:${p.x}
            y: ${p.y} and the z: ${p.z}
            `)

            tempPoses.push(p.x, p.y, p.z)
            logPoseButton.disabled = true
            document.getElementById('logPose').innerText = 'wait for pose to be logged' ;
            setTimeout(() => {

                logPoseButton.disabled = false
                document.getElementById('logPose').innerText = 'Pose is logged add another';
            }, 2000)



    }
    allposes.push({data: tempPoses, label: formField})
    console.log(allposes)
    localStorage.setItem("gym-poses", JSON.stringify(allposes))


}

startApp()
