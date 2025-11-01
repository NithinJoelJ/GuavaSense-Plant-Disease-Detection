import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!image) return alert("Please upload a leaf image first!");

    setLoading(true);
    setResult("");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await axios.post("http://127.0.0.1:5001/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(`${res.data.prediction.class} (${(res.data.prediction.confidence*100).toFixed(2)}% confidence)`);
    } catch (err) {
      console.error(err);
      setResult("Error: Unable to get prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="logo">
          <i className="fa fa-leaf"></i> GuavaSense
        </div>
      </nav>

      <main className="main-section">
        <div className="text-section">
          <h1>Plants make a positive impact on your environment.</h1>
          <p>
            Upload your guava plant leaf to identify diseases instantly with our
            AI-powered detection system.
          </p>

          <div className="buttons">
            <label htmlFor="file-upload" className="upload-btn">
              Upload Image
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              hidden
            />
            <button className="predict-btn" onClick={handlePredict}>
              {loading ? "Predicting..." : "Predict"}
            </button>
          </div>

          {result && (
            <div className="result">
              <strong>Prediction Result:</strong> {result}
            </div>
          )}
        </div>

        <div className="image-section">
          {preview ? (
            <img src={preview} alt="Preview" className="leaf-preview" />
          ) : (
            <div className="placeholder">Your leaf image will appear here</div>
          )}
        </div>
      </main>

      {/* About the Project Section */}
      <section className="about-section">
        <h2>About the Project</h2>
        <p>
          Hi this is Sreeja Bonthu(22MID0193) and C.Jahnavi(22MID0200) <br/>Together, we built this Guava Plant Disease Detection system to help farmers and plant enthusiasts quickly identify diseases in guava leaves.
Simply upload a guava leaf image, and our AI-powered system will detect whether it is Healthy or affected by diseases such as Canker, Leaf Spot, Mummification, or Rust, and show the confidence percentage for the prediction.
Our goal is to provide a simple, fast, and reliable tool that can help monitor plant health and support timely intervention to protect guava crops.</p>
      </section>

      <footer>
        <p>© 2025 GuavaSense | Guava Plant Health Detection</p>
      </footer>
    </div>
  );
}

export default App;