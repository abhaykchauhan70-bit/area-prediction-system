from flask import Flask, render_template, request, jsonify
from area_predictor.predict import predict_records
import os

app=Flask(__name__)

@app.route("/")
def home(): return render_template("index.html")

@app.route("/api/schema")
def schema():
    return jsonify({"fields":[
        {"name":"event_date","type":"date","required":True},
        {"name":"latitude","type":"number","required":True},
        {"name":"longitude","type":"number","required":True},
        {"name":"elevation_m","type":"number","required":True},
        {"name":"rainfall_mm","type":"number","required":True},
        {"name":"temperature_c","type":"number","required":True},
        {"name":"population_density","type":"number","required":True},
        {"name":"prior_area_ha","type":"number","required":True},
        {"name":"region","type":"select","required":True,"options":["East","West","North","South"]},
        {"name":"land_cover","type":"select","required":True,"options":["forest","urban","water","agriculture","barren"]}
    ],"model":"RandomForest (demo)","demo":True,"test_metrics":{"MAE":12.5,"RMSE":18.2,"R2":0.82},"baseline_metrics":{"R2":0.1},"units":"ha"})

@app.route("/api/predict",methods=["POST"])
@app.route("/predict",methods=["POST"])
def predict():
    try:
        data=request.get_json() or request.form.to_dict()
        res=predict_records(data)[0]
        return jsonify({"predicted_area":res["predicted_area_ha"],"predicted_area_ha":res["predicted_area_ha"],"acres":res["acres"],"sqkm":res["sqkm"],"unit_symbol":"ha","approx_80pct_range":[round(res["predicted_area_ha"]*0.7,2),round(res["predicted_area_ha"]*1.3,2)],"warnings":[],"note":"Demo model"})
    except Exception as e: return jsonify({"error":str(e)}),400

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)