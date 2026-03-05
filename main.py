from dotenv import load_dotenv
from agent.graph import build_graph
from datetime import datetime

load_dotenv()

def save_report(report: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/test_report_{timestamp}.txt"
    
    import os
    os.makedirs("reports", exist_ok=True)
    
    with open(filename, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PORSCHE ENGINEERING\n")
        f.write("VEHICLE TEST ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n>> Report saved to {filename}")


def main():
    print("=" * 60)
    print("PORSCHE ENGINEERING — VEHICLE TEST ANALYSIS AGENT")
    print("=" * 60)

    graph = build_graph()

    initial_state = {
        "raw_data": None,
        "anomalies": None,
        "similar_failures": None,
        "final_report": None
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_report"])
    
    save_report(result["final_report"])


if __name__ == "__main__":
    main()