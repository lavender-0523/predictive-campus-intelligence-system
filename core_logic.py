"""
core_logic.py
===============================================================================
Prototype Predictive Campus Intelligence System: Comprehensive Academic Backend
Designed for 2nd-Year Computer Science Undergraduates
Authors: Student 1 (Modules 1, 2, 4) & Student 2 (Modules 3, 5, 6-Regex)

Direct Syllabus Coverage:
    - Module 1: Basic Data Types, Math Functions, Tuples, Lists, Dictionaries, Sets
    - Module 2: If-elif-else, For/While Loops, Break, Try-Except, Inner Functions,
                Lambda, map(), and filter()
    - Module 3: OOP (Classes, Self, Constructors, Inner Classes, Encapsulation,
                Inheritance, super(), Abstract Classes, Method Overriding)
    - Module 4: Exception Handling, Assert Statements, Module Exporting (__all__)
    - Module 5: File Handling with 'with', Pickle Serialization, NumPy, Pandas,
                Scikit-Learn Machine Learning (Regression, Train/Test, Metrics)
    - Module 6: Regular Expressions (re.match), Recommendation System Logic
===============================================================================
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import os
import pickle
import re
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Module 4: Explicit Module Export List
__all__ = [
    "CAMPUS_ZONES",
    "ZONE_METADATA",
    "ZONE_TYPES",
    "HIGH_CONGESTION_THRESHOLD",
    "MODERATE_CONGESTION_THRESHOLD",
    "ZoneNotFoundError",
    "validate_time_format",
    "generate_campus_dataset",
    "classify_risk_tier",
    "filter_congested_zones",
    "enrich_zones_with_map",
    "BasePredictor",
    "CampusPredictor",
    "save_model_to_disk",
    "load_model_from_disk",
    "get_rerouting_recommendations",
]


# =============================================================================
# MODULE 1: BASIC DATA TYPES, STRUCTURES, & MATH UTILITIES
# =============================================================================

# Module 1 Data Structure: List of all 8 monitored campus zones
CAMPUS_ZONES: List[str] = [
    "Canteen",
    "Library",
    "Labs",
    "Lifts",
    "Auditorium",
    "Corridors",
    "Parking",
    "Cafeteria B",
]

# Module 1 Data Structure: Set of unique facility categories
ZONE_TYPES: Set[str] = {"Dining", "Study", "Transit", "Events", "Utility"}

# Numbers in Python (Floats & Integers)
HIGH_CONGESTION_THRESHOLD: float = 75.0
MODERATE_CONGESTION_THRESHOLD: float = 45.0
DEFAULT_CACHE_FILENAME: str = "trained_campus_model.pkl"

# Module 1 Data Structure: Nested Dictionary with Tuples for spatial coordinates (X, Y)
ZONE_METADATA: Dict[str, Dict[str, Any]] = {
    "Canteen": {"coords": (20, 30), "type": "Dining", "capacity": 250, "alternate": "Cafeteria B"},
    "Cafeteria B": {"coords": (80, 25), "type": "Dining", "capacity": 150, "alternate": "Canteen"},
    "Library": {"coords": (35, 80), "type": "Study", "capacity": 300, "alternate": "Labs"},
    "Labs": {"coords": (65, 80), "type": "Study", "capacity": 200, "alternate": "Library"},
    "Lifts": {"coords": (48, 50), "type": "Transit", "capacity": 60, "alternate": "Corridors"},
    "Corridors": {"coords": (52, 50), "type": "Transit", "capacity": 180, "alternate": "Lifts"},
    "Auditorium": {"coords": (15, 65), "type": "Events", "capacity": 500, "alternate": "Corridors"},
    "Parking": {"coords": (85, 60), "type": "Utility", "capacity": 220, "alternate": "Corridors"},
}


# =============================================================================
# MODULE 4 & 6: CUSTOM EXCEPTIONS & REGULAR EXPRESSIONS
# =============================================================================

class ZoneNotFoundError(Exception):
    """Module 4: Custom User-Defined Exception for invalid zone lookups."""
    def __init__(self, zone_name: str):
        super().__init__(f"Error: Campus Zone '{zone_name}' is not recognized in system metadata.")


def validate_time_format(time_string: str) -> bool:
    """
    Module 6 Requirement: Uses Regular Expressions (re.match) to validate HH:MM clock format.
    Example: '13:45' -> True, '25:00' -> False
    """
    pattern = r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$"
    match_result = re.match(pattern, time_string)
    return match_result is not None


# =============================================================================
# MODULE 2 & 5: DATA GENERATION, LOOPS & PANDAS / NUMPY (STUDENT 1)
# =============================================================================

def generate_campus_dataset(num_days: int = 30) -> pd.DataFrame:
    """
    Module 2: Uses For/While Loops, Break statements, If-elif-else decision making.
    Module 4: Uses Assert statements to check parameter validity.
    Module 5: Uses NumPy & Pandas for data manipulation and temporal time-series shifting.
    """
    # Module 4: Assert Statement for defensive parameter checking
    assert isinstance(num_days, int) and num_days > 0, "num_days must be a positive integer"

    np.random.seed(42)  # Ensures 100% reproducible historical data
    start_date = datetime(2026, 1, 1, 8, 0)
    records = []

    # Module 2 Looping: For loop across days
    for day in range(num_days):
        current_day = start_date + timedelta(days=day)
        day_of_week = current_day.weekday()  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        weekend_discount = 0.30 if is_weekend else 1.00

        # Module 2 Looping: While loop with a Break statement
        step_idx = 0
        while True:
            if step_idx >= 44:  # 44 intervals of 15-mins covers 08:00 to 19:00
                break

            dt = current_day + timedelta(minutes=step_idx * 15)
            hour = dt.hour
            minute = dt.minute

            for zone in CAMPUS_ZONES:
                # Base occupancy
                occupancy = 18.0

                # Module 2 Decision Making: If-elif-else conditional logic for peak hours
                if zone in ["Canteen", "Cafeteria B"]:
                    if 12 <= hour <= 13:          # Lunch rush
                        occupancy += 65.0
                    elif hour == 10 and minute >= 15: # Snack rush
                        occupancy += 30.0

                elif zone in ["Library", "Labs"]:
                    if 14 <= hour <= 17:          # Afternoon study peak
                        occupancy += 55.0

                elif zone in ["Lifts", "Corridors"]:
                    if hour in [9, 11, 13, 15] and minute <= 15: # Lecture changeover
                        occupancy += 60.0

                elif zone == "Auditorium":
                    if hour == 15:                # Afternoon guest lecture
                        occupancy += 45.0

                elif zone == "Parking":
                    if hour == 8 or hour == 17:   # Inflow/Outflow rush
                        occupancy += 55.0

                # Add stochastic noise and apply weekend multiplier
                noise = np.random.uniform(-3.5, 3.5)
                occupancy = (occupancy * weekend_discount) + noise

                # Module 1 Built-in Math Functions: min, max, round
                occupancy = max(0.0, min(100.0, occupancy))

                records.append({
                    "timestamp": dt,
                    "zone": zone,
                    "hour": hour,
                    "minute": minute,
                    "day_of_week": day_of_week,
                    "occupancy_pct": round(occupancy, 1),
                })

            step_idx += 1

    # Module 5: Pandas DataFrame manipulation
    df = pd.DataFrame(records)
    df.sort_values(by=["zone", "timestamp"], inplace=True)

    # 30-Minute Forward Target Creation (+2 rows of 15m intervals)
    df["future_occupancy_30m"] = df.groupby("zone")["occupancy_pct"].shift(-2)
    df.dropna(inplace=True)
    return df


# =============================================================================
# MODULE 2: FUNCTIONS, INNER FUNCTIONS, LAMBDA, MAP & FILTER (STUDENT 1)
# =============================================================================

def classify_risk_tier(occupancy_pct: float) -> str:
    """
    Module 2 Function: Describing a function with an inner helper function.
    Demonstrates checking/setting default parameters and inner function calls.
    """
    # Module 2: Function inside a function (Inner Function)
    def evaluate_tier(val: float, high: float = HIGH_CONGESTION_THRESHOLD, mod: float = MODERATE_CONGESTION_THRESHOLD) -> str:
        if val >= high:
            return "HIGH_CONGESTION"
        elif val >= mod:
            return "MODERATE"
        return "NORMAL"

    return evaluate_tier(occupancy_pct)


def filter_congested_zones(zones_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Module 2 Requirement: Uses filter() and a lambda function
    to isolate zones operating at or above the 75% threshold.
    """
    return list(
        filter(
            lambda z: z["predicted_occupancy"] >= HIGH_CONGESTION_THRESHOLD,
            zones_list
        )
    )


def enrich_zones_with_map(zones_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Module 2 Requirement: Uses map() and a lambda function
    to transform raw metrics into formatted telemetry cards.
    """
    badge_lookup = {
        "HIGH_CONGESTION": "🔴 High Congestion",
        "MODERATE": "🟡 Moderate Load",
        "NORMAL": "🟢 Normal Flow",
    }

    return list(
        map(
            lambda z: {
                "zone": z["zone"],
                "current_occupancy": z["current_occupancy"],
                "predicted_occupancy": z["predicted_occupancy"],
                "risk_tier": classify_risk_tier(z["predicted_occupancy"]),
                "badge": badge_lookup[classify_risk_tier(z["predicted_occupancy"])],
                "capacity": ZONE_METADATA[z["zone"]]["capacity"],
                "type": ZONE_METADATA[z["zone"]]["type"],
                "coords": ZONE_METADATA[z["zone"]]["coords"],
            },
            zones_list
        )
    )


# =============================================================================
# MODULE 3: OBJECT-ORIENTED PROGRAMMING (OOP) ARCHITECTURE (STUDENT 2)
# =============================================================================

class BasePredictor(ABC):
    """
    Module 3: Data Abstraction & Interface.
    Abstract Base Class defining the contract for Campus Forecasting Models.
    """

    def __init__(self, model_name: str = "BasePredictor"):
        # Encapsulation: Protected member
        self._model_name = model_name

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Abstract method to train models."""
        pass

    @abstractmethod
    def predict(self, zone: str, hour: int, minute: int, day_of_week: int, current_occ: float) -> float:
        """Abstract method for forward inference."""
        pass


class CampusPredictor(BasePredictor):
    """
    Module 3 OOP Concepts:
        - Inheritance: Inherits from BasePredictor
        - super() Method: Calls parent constructor
        - Encapsulation: Private member (__feature_cols) & Protected member (_models)
        - Inner Class: ModelMetrics to encapsulate performance telemetry
        - Polymorphism & Method Overriding: Overrides train() and predict()
    """

    # Module 3: Inner Class
    class ModelMetrics:
        """Inner Class to package evaluation results per zone."""
        def __init__(self, mae: float, r2: float):
            self.mae = round(mae, 2)
            self.r2 = round(r2, 3)

        def to_dict(self) -> Dict[str, float]:
            return {"MAE (Error %)": self.mae, "R2 Score": self.r2}

    def __init__(self):
        # Module 3: Calling parent constructor using super()
        super().__init__(model_name="CampusRandomForestEnsemble")
        
        # Encapsulation: Protected dictionary for trained models
        self._models: Dict[str, RandomForestRegressor] = {}
        
        # Encapsulation: Private feature list (name mangling: _CampusPredictor__feature_cols)
        self.__feature_cols: List[str] = ["hour", "minute", "day_of_week", "occupancy_pct"]
        
        # Telemetry container storing Inner Class instances
        self.metrics_container: Dict[str, CampusPredictor.ModelMetrics] = {}

    # Module 3: Method Overriding (Polymorphism)
    def train(self, df: pd.DataFrame) -> None:
        """Trains an explainable Random Forest Regressor per zone."""
        for zone in CAMPUS_ZONES:
            zone_df = df[df["zone"] == zone]
            X = zone_df[self.__feature_cols]
            y = zone_df["future_occupancy_30m"]

            # Module 5: Train/Test Split (80/20) with temporal order preserved
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, shuffle=False
            )

            # Lightweight explainable Random Forest
            reg = RandomForestRegressor(n_estimators=35, max_depth=6, random_state=42)
            reg.fit(X_train, y_train)

            # Evaluation
            y_pred = reg.predict(X_test)
            mae_val = mean_absolute_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)

            self._models[zone] = reg
            # Instantiate Inner Class
            self.metrics_container[zone] = CampusPredictor.ModelMetrics(mae=mae_val, r2=r2_val)

    # Module 3: Method Overriding (Polymorphism)
    def predict(self, zone: str, hour: int, minute: int, day_of_week: int, current_occ: float) -> float:
        """Performs 30-minute lookahead inference."""
        # Module 4: Exception Handling (try-except)
        try:
            if zone not in self._models:
                raise ZoneNotFoundError(zone)

            input_vec = pd.DataFrame([{
                "hour": hour,
                "minute": minute,
                "day_of_week": day_of_week,
                "occupancy_pct": current_occ,
            }])

            prediction = self._models[zone].predict(input_vec)[0]
            return float(np.clip(prediction, 0.0, 100.0))

        except ZoneNotFoundError as e:
            # Fallback gracefully if an unexpected zone is requested
            print(f"[Warning caught]: {e}")
            return current_occ


# =============================================================================
# MODULE 5: ADVANCED FILE HANDLING & PICKLE SERIALIZATION (STUDENT 2)
# =============================================================================

def save_model_to_disk(predictor_obj: CampusPredictor, filepath: str = DEFAULT_CACHE_FILENAME) -> bool:
    """
    Module 5 Requirement: Uses 'with' statement and 'pickle' for serialization (dump).
    Saves trained ML pipeline to disk.
    """
    try:
        with open(filepath, "wb") as f:
            pickle.dump(predictor_obj, f)
        return True
    except Exception as err:
        print(f"File Error: Could not serialize model to {filepath}. Detail: {err}")
        return False


def load_model_from_disk(filepath: str = DEFAULT_CACHE_FILENAME) -> CampusPredictor:
    """
    Module 5 Requirement: Uses 'with' statement and 'pickle' for deserialization (load).
    Loads pre-trained ML pipeline from disk if available.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                loaded_model = pickle.load(f)
            return loaded_model
        except Exception as err:
            print(f"File Error: Failed to deserialize {filepath}. Detail: {err}")
    return None


# =============================================================================
# MODULE 6: RECOMMENDATION SYSTEM & MITIGATION LOGIC (STUDENT 2 & 3)
# =============================================================================

def get_rerouting_recommendations(enriched_zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Module 6 Project-Based Requirement: Rule-based recommendation engine.
    Finds forecasted bottlenecks (>=75%) and suggests lower-load alternate zones.
    """
    suggestions = []
    zone_dict = {z["zone"]: z for z in enriched_zones}

    for zone in enriched_zones:
        if zone["risk_tier"] == "HIGH_CONGESTION":
            congested_name = zone["zone"]
            alternate_name = ZONE_METADATA[congested_name]["alternate"]
            alt_zone = zone_dict.get(alternate_name)

            if alt_zone and alt_zone["predicted_occupancy"] < HIGH_CONGESTION_THRESHOLD:
                buffer = zone["predicted_occupancy"] - alt_zone["predicted_occupancy"]
                suggestions.append({
                    "congested_zone": congested_name,
                    "congested_pred": zone["predicted_occupancy"],
                    "suggested_zone": alternate_name,
                    "suggested_pred": alt_zone["predicted_occupancy"],
                    "relief_headroom": round(buffer, 1),
                })

    return suggestions
