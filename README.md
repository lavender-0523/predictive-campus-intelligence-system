# 🏛️ Prototype Predictive Campus Intelligence System
> **Demonstrating Historical Occupancy Modeling for Congestion Mitigation**  
> *Second-Year Computer Science Project (Modules 1–6 Syllabus Aligned)*

---

## 📌 Project Overview
The **Predictive Campus Intelligence System** is an explainable machine learning prototype designed for academic presentation. It models historical foot-traffic trends to forecast crowd density **30 minutes ahead** across 8 key campus zones and dynamically recommends low-congestion routing alternatives.

---

## 🏗️ Architecture & Syllabus Alignment

```
mini_project_ppca_1/
│
├── core_logic.py          # Backend Logic, Data Simulation & ML Engine
├── ui_app.py              # Streamlit Web Dashboard & Plotly 2D Spatial Map
├── requirements.txt       # Dependencies
├── .gitignore             # Git Ignore Configuration
└── README.md              # Project Documentation
```

### 📚 Syllabus Mapping Matrix
| Module | Topics Covered | Implementation Details |
| :--- | :--- | :--- |
| **Module 1** | Python Fundamentals & Data Structures | Tuples `(X, Y)`, Lists, Nested Dictionaries (`ZONE_METADATA`), Sets (`ZONE_TYPES`), Math functions (`min`, `max`, `round`). |
| **Module 2** | Decision Making, Loops & Functions | Conditional `if-elif-else`, `while-break` loops, inner functions (`classify_risk_tier`), anonymous `lambda`, `map()`, and `filter()`. |
| **Module 3** | Object-Oriented Programming (OOP) | `BasePredictor(ABC)` abstract class/interface, `CampusPredictor` derived class, `super()`, encapsulation (private `__feature_cols`, protected `_models`), inner class `ModelMetrics`. |
| **Module 4** | Exception Handling & Packages | Custom exception `ZoneNotFoundError`, defensive `assert` checks, module export `__all__`. |
| **Module 5** | Data Handling & Machine Learning | `with open()` file handling, `pickle` serialization/deserialization, Pandas & NumPy data wrangling, Scikit-Learn `LinearRegression`, chronological train/test splitting (`shuffle=False`), $MAE$ and $R^2$ evaluation. |
| **Module 6** | GUI, Regular Expressions & Recommender | Clean executive white Streamlit dashboard, Plotly 2D spatial density map, Regex time validation (`re.match`), rule-based alternate rerouting engine. |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lavender-0523/predictive-campus-intelligence-system.git
cd predictive-campus-intelligence-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run ui_app.py
```

---

## 📊 Monitored Campus Facilities

1. **Canteen** (Dining) $\rightarrow$ *Alternate: Cafeteria B*
2. **Cafeteria B** (Dining) $\rightarrow$ *Alternate: Canteen*
3. **Library** (Study) $\rightarrow$ *Alternate: Labs*
4. **Labs** (Study) $\rightarrow$ *Alternate: Library*
5. **Lifts** (Transit) $\rightarrow$ *Alternate: Corridors*
6. **Corridors** (Transit) $\rightarrow$ *Alternate: Lifts*
7. **Auditorium** (Events) $\rightarrow$ *Alternate: Corridors*
8. **Parking** (Utility) $\rightarrow$ *Alternate: Corridors*

---

## 👥 Student Presentation Breakdown

* **Student 1 (Modules 1, 2, 4)**: Data generation with diurnal curves, `map()` & `filter()` functional programming, assertion checks.
* **Student 2 (Modules 3, 5)**: Object-Oriented architecture (`BasePredictor` $\rightarrow$ `CampusPredictor`), Scikit-Learn regression, $MAE$/$R^2$ metrics, Pickle serialization.
* **Student 3 (Module 6)**: Streamlit interactive interface, Plotly 2D coordinate campus floorplan, live preset scenarios, proactive rerouting engine.
