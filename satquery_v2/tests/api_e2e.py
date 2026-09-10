import os
os.environ["HF_ENABLED"] = "0"
from fastapi.testclient import TestClient
import app as application

client = TestClient(application.app)

print("health:", client.get("/api/health").json())
status = client.get("/api/model").json()
print("model state:", status["state"], "|", status["message"][:70])
opts = client.get("/api/options").json()
print("modes:", [m["id"] for m in opts["modes"]], "| presets:", len(opts["band_presets"]))
print("index:", client.get("/").status_code, "css:", client.get("/static/css/app.css").status_code,
      "anime:", client.get("/static/js/anime.esm.min.js").status_code,
      "main:", client.get("/static/js/main.js").status_code)

S = "data/sample"
def f(name):
    return (name, open(os.path.join(S, name), "rb"), "image/png")

cases = [
    ("single", {"image_a": f("optical_t1.png")}, "What is present in this image?"),
    ("change", {"image_a": f("optical_t1.png"), "image_b": f("optical_t2.png")},
     "Calculate the percentage of changed area."),
    ("fusion", {"image_a": f("optical_t1.png"), "image_b": f("sar_t1.png")},
     "Compare these optical and SAR images."),
]
for mode, files, q in cases:
    r = client.post("/api/analyze", data={"mode": mode, "query": q, "band_preset": ""}, files=files)
    j = r.json()
    print(f"\n=== {mode} [{r.status_code}] success={j.get('success')} ===")
    print("  headline:", (j.get("answer") or "").splitlines()[0][:110])
    print("  measurements:", len(j.get("measurements") or []),
          "| trace steps:", len(j.get("trace") or []),
          "| evidence:", "yes" if j.get("evidence") else "no",
          "| source:", j.get("model_source"))

# error paths
r = client.post("/api/analyze", data={"mode": "change", "query": "x"},
                files={"image_a": f("optical_t1.png")})
print("\nmissing second image:", r.status_code, "-", r.json()["answer"])
r = client.post("/api/analyze", data={"mode": "single", "query": "x"},
                files={"image_a": ("bad.exe", b"nope", "application/octet-stream")})
print("bad file type:", r.status_code, "-", r.json()["answer"])
r = client.post("/api/analyze", data={"mode": "nonsense", "query": "x"},
                files={"image_a": f("optical_t1.png")})
print("bad mode:", r.status_code, "-", r.json()["answer"][:60])
print("\nALL API CHECKS DONE")
