import pandas as pd
import numpy as np

def generate_sensor_data():
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "speed_kmh": np.random.normal(120, 15, n),
        "brake_pressure_bar": np.random.normal(30, 3, n),
        "engine_temp_c": np.random.normal(90, 5, n),
        "torque_nm": np.random.normal(300, 20, n),
    })

    # Inject anomalies
    df.loc[150, "brake_pressure_bar"] = 72
    df.loc[300, "engine_temp_c"] = 140
    df.loc[420, "torque_nm"] = 600

    return df