"use client";

import { ChangeEvent, useEffect, useState } from "react";

type Prediction = {
  label: string;
  probability: number;
};

type Result = {
  accepted: boolean;
  message?: string;
  predicted?: string;
  confidence?: number;
  low_confidence?: boolean;
  top3?: Prediction[];
  original?: string;
  gradcam?: string;
};

type AttackResult = {
  clean: Prediction;
  adversarial: Prediction;
  image: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function pretty(label: string) {
  return label.replace("___", " · ").replaceAll("_", " ");
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [model, setModel] = useState("mobilenet");
  const [threshold, setThreshold] = useState(0.6);
  const [epsilon, setEpsilon] = useState(0.02);
  const [result, setResult] = useState<Result | null>(null);
  const [attack, setAttack] = useState<AttackResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [attackLoading, setAttackLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setResult(null);
    setAttack(null);
    setError("");
    setPreview(selected ? URL.createObjectURL(selected) : "");
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(
        `${API_URL}/predict?model=${model}&confidence_threshold=${threshold}`,
        { method: "POST", body: form },
      );
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Prediction failed");
      setResult(payload);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  async function runAttack() {
    if (!file) return;
    setAttackLoading(true);
    const form = new FormData();
    form.append("file", file);
    setError("");
    try {
      const response = await fetch(`${API_URL}/attack?model=${model}&epsilon=${epsilon}`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Attack failed");
      setAttack(payload);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Attack failed");
    } finally {
      setAttackLoading(false);
    }
  }

  function changeModel(nextModel: string) {
    setModel(nextModel);
    setResult(null);
    setAttack(null);
    setError("");
  }

  return (
    <main className="shell">
      <header className="hero">
        <div className="eyebrow">INT422 · DEEP LEARNING</div>
        <h1>Leaf<span>Guard</span></h1>
        <p>Understand what your plant is telling you.</p>
        <div className="status"><i /> Explainable AI workspace</div>
      </header>

      <section className="workspace">
        <aside className="controls panel">
          <div className="section-label">01 / MODEL</div>
          <label>
            Inference model
            <select value={model} onChange={(event) => changeModel(event.target.value)}>
              <option value="mobilenet">MobileNetV2 · Transfer learning</option>
              <option value="baseline">LeafGuard CNN · Custom</option>
            </select>
          </label>
          <label>
            Warning threshold <strong>{Math.round(threshold * 100)}%</strong>
            <input type="range" min="0" max="1" step="0.05" value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))} />
          </label>
          <div className="section-label">02 / IMAGE</div>
          <label className="dropzone">
            <input type="file" accept="image/png,image/jpeg" onChange={selectFile} />
            <span className="upload-icon">↑</span>
            <b>{file ? file.name : "Drop a leaf image here"}</b>
            <small>JPG or PNG · Best results with a clear leaf</small>
          </label>
          <button className="primary" disabled={!file || loading} onClick={analyze}>
            {loading ? "Analyzing..." : "Analyze leaf"} <span>→</span>
          </button>
          {error && <p className="error">{error}</p>}
        </aside>

        <section className="results">
          {!result && <div className="empty panel"><div className="empty-mark">✦</div><h2>Your explanation starts here</h2><p>Upload a leaf photo to see the diagnosis, confidence, and visual evidence behind the model&apos;s decision.</p></div>}
          {result && !result.accepted && <div className="panel alert"><h2>Image rejected</h2><p>{result.message}</p></div>}
          {result?.accepted && (
            <>
              <div className="diagnosis panel">
                <div><div className="section-label">PREDICTED DISEASE</div><h2>{pretty(result.predicted ?? "")}</h2></div>
                <div className="confidence"><strong>{((result.confidence ?? 0) * 100).toFixed(1)}%</strong><span>confidence</span></div>
              </div>
              {result.low_confidence && <div className="warning">Low confidence — use this result as a screening aid, not a diagnosis.</div>}
              <div className="visuals">
                <div className="image-card panel"><div className="card-title">Original image</div><img src={result.original} alt="Uploaded leaf" /></div>
                <div className="image-card panel"><div className="card-title">Grad-CAM evidence <span>Warm areas influenced the prediction</span></div><img src={result.gradcam} alt="Grad-CAM heatmap overlay" /></div>
              </div>
              <div className="panel ranking"><div className="card-title">Top 3 possibilities</div>{result.top3?.map((item) => <div className="rank" key={item.label}><span>{pretty(item.label)}</span><div><div className="bar"><i style={{ width: `${item.probability * 100}%` }} /></div><b>{(item.probability * 100).toFixed(1)}%</b></div></div>)}</div>
              <div className="panel attack"><div className="card-title">Adversarial robustness <span>FGSM demonstration</span></div><p>See how a tiny, intentional perturbation can change the model&apos;s confidence.</p><div className="attack-controls"><input type="range" min="0" max="0.1" step="0.005" value={epsilon} onChange={(event) => setEpsilon(Number(event.target.value))} /><b>ε {epsilon.toFixed(3)}</b><button disabled={attackLoading} onClick={runAttack}>{attackLoading ? "Running..." : "Run attack"}</button></div>{attack && <div className="attack-output"><img src={attack.image} alt="FGSM adversarial image" /><span>Clean: <b>{pretty(attack.clean.label)}</b><br />FGSM: <b>{pretty(attack.adversarial.label)}</b></span></div>}</div>
            </>
          )}
        </section>
      </section>
      <footer>LeafGuard <span>·</span> Model explanations are generated with Grad-CAM</footer>
    </main>
  );
}
