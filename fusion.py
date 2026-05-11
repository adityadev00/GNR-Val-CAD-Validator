
import adsk.core
import adsk.fusion
import traceback
import subprocess
import tempfile
import json
import os

# ============================================================
# EDIT THESE 2 PATHS ONLY
# ============================================================
PYTHON_EXE = r"C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe"
APP_PY_PATH = r"C:\Users\YourName\Desktop\GNRVAL\app.py"

# Default metadata for MVP
DEFAULT_MATERIAL = "steel"
DEFAULT_TYPE = "housing"
DEFAULT_TOLERANCE = 0.10
DEFAULT_ROUGHNESS = 1.6


def _get_app():
    return adsk.core.Application.get()


def _get_ui():
    app = _get_app()
    return app.userInterface if app else None


def _mm_from_internal(design, value):
    try:
        um = design.unitsManager
        return float(um.convert(value, um.internalUnits, "mm"))
    except:
        # fallback
        return float(value)


def _pick_body_from_active_design(design):
    root = design.rootComponent

    # 1) If a BRepBody is selected, use that first
    app = _get_app()
    ui = _get_ui()

    try:
        sels = ui.activeSelections
        for i in range(sels.count):
            ent = sels.item(i).entity
            body = adsk.fusion.BRepBody.cast(ent)
            if body:
                return body
    except:
        pass

    # 2) Otherwise first root body
    if root.bRepBodies.count > 0:
        return root.bRepBodies.item(0)

    # 3) Otherwise first body from first occurrence/component that has one
    for occ in root.allOccurrences:
        comp = occ.component
        if comp and comp.bRepBodies.count > 0:
            return comp.bRepBodies.item(0)

    return None


def _body_to_component_dict(design, body):
    bbox = body.boundingBox
    min_pt = bbox.minPoint
    max_pt = bbox.maxPoint

    x_mm = _mm_from_internal(design, abs(max_pt.x - min_pt.x))
    y_mm = _mm_from_internal(design, abs(max_pt.y - min_pt.y))
    z_mm = _mm_from_internal(design, abs(max_pt.z - min_pt.z))

    # keep actual == nominal for MVP
    component = {
        "actual_x": round(max(1.0, x_mm), 3),
        "actual_y": round(max(1.0, y_mm), 3),
        "actual_z": round(max(1.0, z_mm), 3),
        "tolerance": DEFAULT_TOLERANCE,
        "material": DEFAULT_MATERIAL,
        "type": DEFAULT_TYPE,
        "roughness": DEFAULT_ROUGHNESS,
        "nominal_dims": [
            round(max(1.0, x_mm), 3),
            round(max(1.0, y_mm), 3),
            round(max(1.0, z_mm), 3),
        ],
    }
    return component


def _format_result(result):
    if not isinstance(result, dict):
        return "Invalid result returned from backend."

    if "error" in result:
        return f"Error: {result['error']}"

    scores = result.get("scores", {})
    verdict = scores.get("verdict", "Unknown")
    fused = scores.get("fused_score", "N/A")
    ml_conf = result.get("ml_confidence", "N/A")
    n_viol = result.get("n_violations", 0)

    top_reasons = result.get("top_reasons", [])
    reason_lines = []
    for r in top_reasons[:3]:
        if isinstance(r, dict):
            sev = r.get("severity", "INFO")
            msg = r.get("message", "No message")
            rule_id = r.get("rule_id", "RULE")
            reason_lines.append(f"- [{sev}] {rule_id}: {msg}")
        else:
            reason_lines.append(f"- {str(r)}")

    if not reason_lines:
        reason_lines.append("- No major reasons returned.")

    text = (
        f"GNR-Val Fusion Validation\n\n"
        f"Verdict: {verdict}\n"
        f"Fused Risk Score: {fused}\n"
        f"ML Confidence: {ml_conf}%\n"
        f"Violations: {n_viol}\n\n"
        f"Top Reasons:\n" + "\n".join(reason_lines)
    )
    return text


def _run_backend_validation(component):
    payload = {"components": [component]}

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "fusion_input.json")
        output_path = os.path.join(tmpdir, "fusion_output.json")

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        cmd = [
            PYTHON_EXE,
            APP_PY_PATH,
            "--fusion-cli",
            input_path,
            output_path,
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # allow first-time training/load
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "Backend validation failed.\n\n"
                f"STDOUT:\n{proc.stdout}\n\n"
                f"STDERR:\n{proc.stderr}"
            )

        if not os.path.exists(output_path):
            raise RuntimeError("Backend finished but output JSON was not created.")

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

    return result


def run(context):
    ui = None
    try:
        app = _get_app()
        ui = _get_ui()

        if not os.path.exists(PYTHON_EXE):
            ui.messageBox(f"Python executable not found:\n{PYTHON_EXE}")
            return

        if not os.path.exists(APP_PY_PATH):
            ui.messageBox(f"app.py not found:\n{APP_PY_PATH}")
            return

        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("No active Fusion 360 design found.")
            return

        body = _pick_body_from_active_design(design)
        if not body:
            ui.messageBox("No solid body found. Select a body or open a model with at least one body.")
            return

        component = _body_to_component_dict(design, body)

        ui.messageBox(
            "Running GNR-Val backend...\n"
            "First run may take longer because model load/train can happen there."
        )

        result = _run_backend_validation(component)
        ui.messageBox(_format_result(result))

    except Exception as e:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))


def stop(context):
    pass
