"""Generate two fake AUTOSAR-style HLD PDFs (v1.0, v1.1) + ground-truth JSON.

Planted inconsistencies (present in both versions):
  R001  DiagnosticHandler requires IDoorLockDiag but no component provides it
  R002  Signal DiagSessionState has a sender but no receiver
  R003  LightingController depends on AmbientSensorDriver, which is never defined
v1.1 changes: +RainSensorHandler (with port/interface/signal/dependency),
              signal VehicleSpeed renamed VehSpeed, port WindowController.PowerStatePort removed.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

SPEC_V10 = {
    "title": "Body Control Module - High-Level Design",
    "components": [
        ("DoorLockManager", "Locks and unlocks the vehicle doors based on vehicle speed and remote requests."),
        ("WindowController", "Controls power window movement and anti-pinch protection."),
        ("LightingController", "Manages exterior and interior lighting based on light requests."),
        ("PowerManager", "Distributes the vehicle power state to all body components."),
        ("DiagnosticHandler", "Handles UDS diagnostic requests and routes them to body components."),
        ("ComSignalRouter", "Routes signals between the body components and the vehicle network."),
    ],
    "interfaces": [
        ("IDoorLock", "Lock and unlock commands for the door actuators."),
        ("IWindowCtrl", "Window movement commands."),
        ("ILightCtrl", "Light switching commands."),
        ("IPowerState", "Current vehicle power state notifications."),
        ("IDiagService", "Diagnostic service request interface."),
        ("ISignalRoute", "Signal routing interface towards the network stack."),
    ],
    # (component, port, direction, interface)
    "ports": [
        ("DoorLockManager", "DoorLockCmd", "provided", "IDoorLock"),
        ("DoorLockManager", "PowerStatePort", "required", "IPowerState"),
        ("WindowController", "WindowCmd", "provided", "IWindowCtrl"),
        ("WindowController", "PowerStatePort", "required", "IPowerState"),
        ("LightingController", "LightCmd", "provided", "ILightCtrl"),
        ("LightingController", "PowerStatePort", "required", "IPowerState"),
        ("PowerManager", "PowerStateOut", "provided", "IPowerState"),
        ("DiagnosticHandler", "DiagReq", "provided", "IDiagService"),
        ("DiagnosticHandler", "DoorLockDiag", "required", "IDoorLockDiag"),
        ("ComSignalRouter", "SignalOut", "provided", "ISignalRoute"),
    ],
    # (name, sender, receiver, cycle)
    "signals": [
        ("DoorLockStatus", "DoorLockManager", "ComSignalRouter", "100 ms"),
        ("WindowPosition", "WindowController", "ComSignalRouter", "100 ms"),
        ("VehicleSpeed", "ComSignalRouter", "DoorLockManager", "20 ms"),
        ("IgnitionState", "PowerManager", "ComSignalRouter", "50 ms"),
        ("LightRequest", "ComSignalRouter", "LightingController", "50 ms"),
        ("DiagSessionState", "DiagnosticHandler", None, "event"),
    ],
    # (from, to, type)
    "dependencies": [
        ("DoorLockManager", "PowerManager", "service"),
        ("WindowController", "PowerManager", "service"),
        ("LightingController", "PowerManager", "service"),
        ("DoorLockManager", "ComSignalRouter", "data"),
        ("DiagnosticHandler", "DoorLockManager", "diagnostic"),
        ("LightingController", "AmbientSensorDriver", "data"),
    ],
    "revisions": [("1.0", "2026-01-10", "Initial release")],
    "planted": [
        {"rule": "R001", "entity": "DiagnosticHandler.DoorLockDiag"},
        {"rule": "R002", "entity": "DiagSessionState"},
        {"rule": "R003", "entity": "LightingController->AmbientSensorDriver"},
    ],
}


def spec_v11() -> dict:
    s = copy.deepcopy(SPEC_V10)
    s["components"].append(("RainSensorHandler", "Reads the rain sensor and publishes the rain intensity."))
    s["interfaces"].append(("IRainStatus", "Rain intensity notifications."))
    s["ports"] = [p for p in s["ports"] if not (p[0] == "WindowController" and p[1] == "PowerStatePort")]
    s["ports"].append(("RainSensorHandler", "RainStatusOut", "provided", "IRainStatus"))
    s["signals"] = [("VehSpeed", *x[1:]) if x[0] == "VehicleSpeed" else x for x in s["signals"]]
    s["signals"].append(("RainIntensity", "RainSensorHandler", "WindowController", "50 ms"))
    s["dependencies"].append(("WindowController", "RainSensorHandler", "data"))
    s["revisions"].append(("1.1", "2026-03-02", "Added RainSensorHandler; renamed VehicleSpeed; updated ports."))
    return s


def ground_truth(spec: dict) -> dict:
    return {
        "components": [c[0] for c in spec["components"]],
        "interfaces": [i[0] for i in spec["interfaces"]],
        "ports": [dict(zip(("component", "name", "direction", "interface"), p)) for p in spec["ports"]],
        "signals": [{"name": s[0], "sender": s[1], "receiver": s[2] or ""} for s in spec["signals"]],
        "dependencies": [dict(zip(("from", "to", "type"), d)) for d in spec["dependencies"]],
        "planted_inconsistencies": spec["planted"],
    }


def _table(rows: list[list[str]], widths: list[float]) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_pdf(spec: dict, version: str, out: Path) -> Path:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=10, leading=14, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=16, leading=20, spaceBefore=8, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=8, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=15, spaceBefore=6, spaceAfter=4)
    title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=22, leading=28, spaceAfter=4)
    P = lambda t, s=body: Paragraph(t, s)  # noqa: E731
    comps = spec["components"]
    names = ", ".join(c[0] for c in comps)

    st = [
        P(spec["title"], title), P(f"Document version {version}", body), Spacer(1, 10),
        P("1 Introduction", h1),
        P("1.1 Purpose", h2),
        P("The purpose of this document is to describe the high-level software architecture of the Body Control Module "
          "(BCM), including its software components, ports, interfaces, signals and dependencies."),
        P("1.2 Scope", h2),
        P("This document covers the application layer components of the BCM. Basic software configuration "
          "is described in separate documents."),
        PageBreak(),
        P("2 System Overview", h1),
        P("2.1 Architecture Overview", h2),
        P(f"The BCM application layer consists of the following software components: {names}. "
          "Components communicate through AUTOSAR ports typed by interfaces, and exchange vehicle signals "
          "through the ComSignalRouter."),
        P("2.2 Component Catalogue", h2),
        _table([["Component", "Description"]] + [[c[0], c[1]] for c in comps], [130, 340]),
        PageBreak(),
        P("3 Software Architecture", h1),
        P("3.1 Interfaces", h2),
        P("The following interfaces are defined for the BCM application layer."),
        _table([["Interface", "Description"]] + [[i[0], i[1]] for i in spec["interfaces"]], [130, 340]),
        PageBreak(),
        P("3.2 Ports", h2),
        P("Each component exposes provided and required ports. Required ports must be connected to a matching provided port."),
        _table([["Component", "Port", "Direction", "Interface"]] + [list(p) for p in spec["ports"]], [120, 120, 80, 120]),
        P("3.2.1 Power State Handling", h3),
        P("The PowerManager provides the power state through the IPowerState interface. DoorLockManager, "
          "LightingController and WindowController require the power state to enter low-power mode safely."),
        PageBreak(),
        P("4 Communication", h1),
        P("4.1 Signal List", h2),
        _table([["Signal", "Sender", "Receiver", "Cycle Time"]] + [[s[0], s[1], s[2] or "-", s[3]] for s in spec["signals"]],
               [120, 120, 120, 80]),
        P("4.2 Component Dependencies", h2),
        _table([["Component", "Depends On", "Type"]] + [list(d) for d in spec["dependencies"]], [150, 150, 100]),
        PageBreak(),
        P("5 Diagnostics", h1),
        P("The DiagnosticHandler receives UDS requests over the IDiagService interface. Requests addressing door "
          "locks are forwarded to the DoorLockManager through the DoorLockDiag port. The current diagnostic "
          "session is published as the DiagSessionState signal."),
        PageBreak(),
        P("6 Revision History", h1),
        _table([["Version", "Date", "Change"]] + [list(r) for r in spec["revisions"]], [70, 90, 310]),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=A4, title=spec["title"], leftMargin=60, rightMargin=60,
                      topMargin=60, bottomMargin=60).build(st)
    return out


def generate(out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir or ROOT / "data")
    paths = {
        "v1.0": build_pdf(SPEC_V10, "1.0", out_dir / "sample_hld_v1.0.pdf"),
        "v1.1": build_pdf(spec_v11(), "1.1", out_dir / "sample_hld_v1.1.pdf"),
    }
    for ver, spec in (("1.0", SPEC_V10), ("1.1", spec_v11())):
        (out_dir / f"ground_truth_v{ver}.json").write_text(json.dumps(ground_truth(spec), indent=2))
    return paths


if __name__ == "__main__":
    for k, v in generate().items():
        print(f"{k}: {v}")
