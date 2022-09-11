from flask import Flask, render_template, request, redirect
import tensorflow as tf
import librosa
import numpy as np

app = Flask(__name__)

def audio_features(file_name):
    # load the audio file
    audio_data, sample_rate = librosa.load(file_name)
    # get the feature
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    # scale the features
    feature_scaled = np.mean(feature.T, axis=0)
    # array of features
    prediction_feature = np.array([feature_scaled])
    return prediction_feature

@app.route("/", methods=["GET", "POST"])
def index():

    transcript = {}
    final = ""
    if request.method == "POST":
        # print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        # file = request.files["file"]
        audio = request.files['file']
        # print(audio)
        interpreter = tf.lite.Interpreter(model_path='ann.tflite')
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # print("Input Shape : ", input_details[0]['shape'])
        # print("Input Tyep : ", input_details[0]['dtype'])
        # print("Output Shape : ", output_details[0]['shape'])
        # print("Output Type : ", output_details[0]['dtype'])

        feas = audio_features(audio)
        # print(feas.shape)

        input_shape = input_details[0]['shape']
        input_data = feas.astype(np.float32)
        # print(input_data.shape)

        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        c_names = ["AC compressor and air filter assembler", "AC noise", "Air filter damage hose",
                   "Alternator output not coming", "Injector not working", "Injector,spares fail/wiring cuts",
                   "Nossil Diesel flow stops", "Nossil noise/injector noise", "Timing chain noise",
                   "Turbo problem", "Water leakage", "brake switch fault",
                   "clipping loose", "dissel injector noise", "no problem",
                   "starter motor fault", "steering sticky noise", "wall tappet clearance"
                   ]

        # yvals = output_data[0]
        # preds = np.argmax(yvals, axis=-1)
        #
        # # print(preds)
        # pred_class = c_names[preds]
        # transcript = pred_class
        # # print(pred_class)
        #
        # # yvals = output_data[0]
        # # # print(yvals)
        # # preds = np.max(yvals)
        # #
        # # if (preds >= 8.0):
        # #     preds = np.argmax(yvals, axis=-1)
        # #     pred_class = c_names[preds]
        # #     transcript = pred_class
        # # else:
        # #     transcript = "Incorrect Output"

        yvals = output_data[0]
        # print(yvals)
        prob = {}
        pro = []

        #here we have considered threshold as 6.0
        for i in range(len(yvals)):
            x = float(str(yvals[i]).split('e', 1)[0])
            # print(x)
            if (x * 10 >= 9.0 and x * 10 < 10.0):
                prob[i] = x * 100
                pro.append(x*100)

            if (x >= 6.0):
                prob[i] = x*10
                pro.append(x*10)

            else :
                pro.append(x)

        # print(prob)

        out = {}

        for keys, values in prob.items():
            pred_class = c_names[keys]
            out[pred_class] = values

        transcript = out
        final = c_names[np.argmax(pro, axis=-1)]
        
        # print(out)

    return render_template('index.html', transcript=transcript, final=final)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
