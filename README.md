---
title: DataWhisperer Pro
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.1
app_file: main.py
pinned: false
---

# DataWhisperer Pro 🎯📊

**DataWhisperer Pro** is an advanced AI-powered data analytics platform that transforms how you explore and understand your datasets. Upload CSV files and unlock intelligent insights through automated EDA, 3D visualizations, AutoML, and natural language AI assistance — all in one comprehensive interface!

## 🚀 Features

### 📊 **Smart EDA (Exploratory Data Analysis)**
- 🔥 **Auto-correlation matrices** with smart constant column handling
- 📈 **Distribution analysis** across multiple numeric variables
- 📦 **Outlier detection** using statistical methods
- 🎨 **Interactive visualizations** powered by Plotly

### 🌌 **3D Data Universe**
- 🌟 **Multi-dimensional scatter plots** with clustering
- 🧬 **PCA-based 3D exploration** with K-means clustering
- 🏔️ **3D surface plots** for relationship mapping
- ⚡ **Performance-optimized** for large datasets

### 🤖 **Secure AutoML Studio**
- 🔒 **Data leakage prevention** with proper train/test splitting
- 🎯 **Auto-classification/regression** detection
- 📊 **Feature importance analysis** with visual rankings
- 🧠 **AI-powered insights** about model performance

### 💬 **AI Assistant (Gemini-Powered)**
- 🤔 **Natural language queries** about your data
- 📝 **Intelligent recommendations** for analysis strategies
- 🎯 **Context-aware responses** based on dataset characteristics
- 💡 **Follow-up suggestions** for deeper exploration

### 📋 **Data Explorer**
- 🔍 **Paginated data viewing** for large datasets
- 📊 **Real-time statistics** and data quality metrics
- 🏷️ **Auto-type detection** (dates, numbers, categories)
- 📈 **Memory usage tracking** and optimization

## 🛠️ Tech Stack

- **Frontend**: Streamlit with responsive design
- **AI Integration**: Google Gemini 1.5 Flash
- **ML/Analytics**: Scikit-learn, Pandas, NumPy
- **Visualizations**: Plotly Express & Graph Objects
- **Data Processing**: Advanced type detection & validation
- **Deployment**: Optimized for Hugging Face Spaces

## 📂 Project Structure

```
DataWhisperer-Pro/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This documentation
└── .streamlit/
    └── secrets.toml       # Streamlit secrets (optional)
```

## ⚡ Quick Start

### 1. **Local Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/DataWhisperer-Pro.git
cd DataWhisperer-Pro

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### 2. **Environment Setup**

Create a `.env` file for local development:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=alternative_key_name
```

### 3. **Hugging Face Deployment**

For Hugging Face Spaces deployment:

1. Fork this repository
2. Create a new Hugging Face Space
3. Add your `GEMINI_API_KEY` in **Settings > Repository secrets**
4. Deploy automatically via git integration

## 🔧 Configuration

### **API Keys Setup**

The application supports multiple API key configurations:

| Method | Priority | Description |
|--------|----------|-------------|
| `GEMINI_API_KEY` | 1 | Primary Gemini API key |
| `GOOGLE_API_KEY` | 2 | Alternative Google API key |
| Streamlit Secrets | 3 | For deployed applications |

### **Performance Tuning**

Adjust these parameters in `main.py` for your deployment:

```python
# Data sampling limits
SAMPLE_SIZE_EDA = 1000      # EDA analysis sample size
SAMPLE_SIZE_3D = 5000       # 3D visualization sample size
MAX_FEATURES = 6            # Maximum features for correlation
MAX_ML_ROWS = 10000         # AutoML training sample size
```

## 📊 Supported Data Formats

- ✅ **CSV files** (UTF-8, Latin1, CP1252 encoding)
- ✅ **Numeric data** (integers, floats)
- ✅ **Categorical data** (strings, objects)
- ✅ **Date/DateTime** (auto-detection and conversion)
- ✅ **Mixed datasets** with automatic type inference

## 🎯 Use Cases

### **Business Analytics**
- 📈 Sales data exploration and trend analysis
- 👥 Customer segmentation and behavior analysis
- 💰 Financial performance monitoring

### **Research & Science**
- 🧪 Experimental data analysis
- 📊 Survey and questionnaire analysis
- 🔬 Hypothesis testing and validation

### **Education**
- 📚 Teaching data science concepts
- 🎓 Student project analysis
- 📋 Assignment and assessment data

## ✨ Demo

Try it live on Hugging Face Spaces:
👉 **[DataWhisperer Pro Live Demo](https://huggingface.co/spaces/Dhruv-18/DataWhisperer-Pro)**

## 🔍 Example Workflows

### **1. Quick Data Overview**
```
1. Upload CSV → 2. View Smart EDA → 3. Check Data Quality → 4. Explore Correlations
```

### **2. Advanced Analysis**
```
1. Upload Dataset → 2. Generate 3D Universe → 3. Run AutoML → 4. Ask AI Questions
```

### **3. Business Intelligence**
```
1. Load Business Data → 2. Identify Key Patterns → 3. Train Predictive Model → 4. Get Strategic Insights
```

## 🚦 Troubleshooting

### **Common Issues**

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY not found` | Add API key to environment or Streamlit secrets |
| `Slow processing on large files` | App auto-samples large datasets for performance |
| `Charts not loading` | Check browser console, try refreshing the page |
| `Upload errors` | Verify CSV format and file encoding |

### **Performance Tips**

- 📊 For datasets >10K rows, the app automatically samples data
- 🔄 Use the "Retry API" button if Gemini connection fails
- 💾 Clear browser cache if visualizations appear corrupted
- ⚡ Close unused tabs to free up memory resources

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📋 Requirements

```txt
streamlit>=1.28.1
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
```

## 🏆 Key Improvements Over V1

- 🔒 **Security**: Fixed data leakage in AutoML pipeline
- ⚡ **Performance**: 3x faster processing with smart sampling
- 🎨 **Visualization**: Enhanced 3D universe with clustering
- 🤖 **AI Integration**: Robust Gemini API with fallbacks
- 📊 **Analytics**: Advanced correlation and outlier detection
- 🛡️ **Reliability**: Comprehensive error handling and validation

## 👨‍💻 Author

**Dhruv Pawar**  
🔗 [GitHub](https://github.com/Dhruv-18) | [Hugging Face](https://huggingface.co/Dhruv-18) | [LinkedIn](https://www.linkedin.com/in/dhruv-pawar-bb8685261/)

*Built with ❤️ during AI/ML internship • Powered by Google Gemini*

## 📜 License

MIT License - see [LICENSE](LICENSE) for details

## 🌟 Acknowledgments

- **Google Gemini** for AI capabilities
- **Streamlit** for the amazing framework
- **Plotly** for interactive visualizations
- **Scikit-learn** for machine learning tools
- **Hugging Face** for seamless deployment platform

---

⭐ **Star this repo** if DataWhisperer Pro helped you unlock insights from your data!