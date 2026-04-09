#!/usr/bin/env python3
"""Check URDF RL-readiness for the "Big Three" quality gates.

The checks are designed for MuJoCo/MJLAB locomotion pipelines:
1) Collision vs visual geometry complexity (performance risk)
2) Inertial mass/inertia sanity (simulation stability risk)
3) Joint limits completeness/range sanity (control realism risk)

Usage:
  python3 check_urdf_big_three.py Sesame.SLDASM.urdf

Exit codes:
  0 -> no failures (warnings may still exist)
  1 -> one or more hard failures detected
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


EPS = 1e-12


@dataclass
class Finding:
    severity: str  # "FAIL" or "WARN"
    category: str
    target: str
    message: str


def as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def resolve_mesh_path(urdf_path: Path, mesh_uri: str) -> Optional[Path]:
    mesh_uri = mesh_uri.strip()
    if not mesh_uri:
        return None

    if mesh_uri.startswith("package://"):
        rel = mesh_uri[len("package://") :]
        return (urdf_path.parent / rel).resolve()
    if mesh_uri.startswith("file://"):
        return Path(mesh_uri[len("file://") :]).resolve()
    return (urdf_path.parent / mesh_uri).resolve()


def count_stl_triangles(path: Path) -> Optional[int]:
    if not path.exists() or path.suffix.lower() != ".stl":
        return None

    try:
        with path.open("rb") as f:
            header = f.read(80)
            if len(header) < 80:
                return None
            tri_bytes = f.read(4)
            if len(tri_bytes) < 4:
                return None
            tri_count = struct.unpack("<I", tri_bytes)[0]

            expected_size = 84 + tri_count * 50
            actual_size = path.stat().st_size
            if expected_size == actual_size:
                return tri_count
    except OSError:
        return None

    # Fallback for ASCII STL: count lines that start with "facet normal".
    try:
        count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lstrip().lower().startswith("facet normal"):
                    count += 1
        if count > 0:
            return count
    except OSError:
        return None

    return None


def extract_geometry_types(element: ET.Element) -> Set[str]:
    geom = element.find("geometry")
    if geom is None:
        return set()

    kinds = set()
    for child in geom:
        kinds.add(child.tag)
    return kinds


def mesh_filenames_from_section(section: ET.Element) -> List[str]:
    geom = section.find("geometry")
    if geom is None:
        return []
    filenames = []
    for mesh in geom.findall("mesh"):
        filename = mesh.get("filename")
        if filename:
            filenames.append(filename)
    return filenames


def check_collision_vs_visual(
    urdf_path: Path,
    root: ET.Element,
    mesh_warn_triangles: int,
    mesh_fail_triangles: int,
) -> List[Finding]:
    findings: List[Finding] = []

    for link in root.findall("link"):
        link_name = link.get("name", "<unnamed_link>")
        visuals = link.findall("visual")
        collisions = link.findall("collision")

        visual_meshes: Set[str] = set()
        for visual in visuals:
            for mesh in mesh_filenames_from_section(visual):
                visual_meshes.add(mesh)

        if not collisions:
            findings.append(
                Finding(
                    "WARN",
                    "CollisionVsVisual",
                    link_name,
                    "Link has no <collision> block; contact behavior may be unrealistic.",
                )
            )
            continue

        for collision in collisions:
            geom_types = extract_geometry_types(collision)
            if not geom_types:
                findings.append(
                    Finding(
                        "WARN",
                        "CollisionVsVisual",
                        link_name,
                        "Collision block has no geometry.",
                    )
                )
                continue

            if "mesh" in geom_types:
                findings.append(
                    Finding(
                        "WARN",
                        "CollisionVsVisual",
                        link_name,
                        "Collision uses <mesh>; consider primitive collision geometry for faster simulation.",
                    )
                )

            for mesh_uri in mesh_filenames_from_section(collision):
                if mesh_uri in visual_meshes:
                    findings.append(
                        Finding(
                            "FAIL",
                            "CollisionVsVisual",
                            link_name,
                            f"Same mesh used for visual and collision: {mesh_uri}",
                        )
                    )

                mesh_path = resolve_mesh_path(urdf_path, mesh_uri)
                if mesh_path is None:
                    continue
                tri_count = count_stl_triangles(mesh_path)
                if tri_count is None:
                    continue

                if tri_count >= mesh_fail_triangles:
                    findings.append(
                        Finding(
                            "FAIL",
                            "CollisionVsVisual",
                            link_name,
                            f"Collision mesh {mesh_path.name} has {tri_count} triangles (>= {mesh_fail_triangles}).",
                        )
                    )
                elif tri_count >= mesh_warn_triangles:
                    findings.append(
                        Finding(
                            "WARN",
                            "CollisionVsVisual",
                            link_name,
                            f"Collision mesh {mesh_path.name} has {tri_count} triangles (>= {mesh_warn_triangles}).",
                        )
                    )

    return findings


def check_inertial_mass(root: ET.Element) -> List[Finding]:
    findings: List[Finding] = []

    for link in root.findall("link"):
        link_name = link.get("name", "<unnamed_link>")
        inertial = link.find("inertial")
        if inertial is None:
            findings.append(
                Finding(
                    "FAIL",
                    "InertiaMass",
                    link_name,
                    "Missing <inertial> block.",
                )
            )
            continue

        mass_elem = inertial.find("mass")
        mass_val = as_float(None if mass_elem is None else mass_elem.get("value"))
        if mass_val is None:
            findings.append(
                Finding("FAIL", "InertiaMass", link_name, "Missing or invalid mass value.")
            )
        elif mass_val <= 0:
            findings.append(
                Finding("FAIL", "InertiaMass", link_name, f"Non-positive mass: {mass_val}")
            )
        elif mass_val < 0.05:
            findings.append(
                Finding(
                    "WARN",
                    "InertiaMass",
                    link_name,
                    f"Very low mass {mass_val:g} kg; verify this is intentional for hardware realism.",
                )
            )

        inertia_elem = inertial.find("inertia")
        if inertia_elem is None:
            findings.append(
                Finding("FAIL", "InertiaMass", link_name, "Missing <inertia> tensor values.")
            )
            continue

        keys = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
        vals = {k: as_float(inertia_elem.get(k)) for k in keys}
        missing = [k for k, v in vals.items() if v is None]
        if missing:
            findings.append(
                Finding(
                    "FAIL",
                    "InertiaMass",
                    link_name,
                    f"Missing/invalid inertia components: {', '.join(missing)}",
                )
            )
            continue

        all_vals = list(vals.values())
        if all(abs(v) <= EPS for v in all_vals):
            findings.append(
                Finding(
                    "FAIL",
                    "InertiaMass",
                    link_name,
                    "All inertia components are zero.",
                )
            )

        if all(abs(v - 1.0) <= 1e-9 for v in all_vals):
            findings.append(
                Finding(
                    "WARN",
                    "InertiaMass",
                    link_name,
                    "All inertia components are 1.0 (likely placeholder values).",
                )
            )

        if vals["ixx"] <= 0 or vals["iyy"] <= 0 or vals["izz"] <= 0:
            findings.append(
                Finding(
                    "FAIL",
                    "InertiaMass",
                    link_name,
                    "Principal inertia terms (ixx, iyy, izz) must be > 0.",
                )
            )

    return findings


def check_joint_limits(root: ET.Element, enforce_servo_180: bool) -> List[Finding]:
    findings: List[Finding] = []

    for joint in root.findall("joint"):
        joint_name = joint.get("name", "<unnamed_joint>")
        joint_type = (joint.get("type") or "").strip().lower()

        if joint_type in {"fixed", "floating"}:
            continue

        limit = joint.find("limit")
        if limit is None:
            findings.append(
                Finding("FAIL", "JointLimits", joint_name, "Missing <limit> block.")
            )
            continue

        effort = as_float(limit.get("effort"))
        velocity = as_float(limit.get("velocity"))
        lower = as_float(limit.get("lower"))
        upper = as_float(limit.get("upper"))

        if effort is None or effort <= 0:
            findings.append(
                Finding("FAIL", "JointLimits", joint_name, "Missing/invalid positive effort.")
            )
        if velocity is None or velocity <= 0:
            findings.append(
                Finding(
                    "FAIL", "JointLimits", joint_name, "Missing/invalid positive velocity."
                )
            )

        if joint_type == "continuous":
            continue

        if lower is None or upper is None:
            findings.append(
                Finding(
                    "FAIL",
                    "JointLimits",
                    joint_name,
                    "Non-continuous joint requires lower and upper limits.",
                )
            )
            continue

        if lower >= upper:
            findings.append(
                Finding(
                    "FAIL",
                    "JointLimits",
                    joint_name,
                    f"Invalid limit range: lower={lower:g}, upper={upper:g}",
                )
            )

        if enforce_servo_180:
            # MG90S style check: range should not exceed ~180 deg (pi rad).
            span = upper - lower
            if span > 3.141592653589793 + 1e-6:
                findings.append(
                    Finding(
                        "WARN",
                        "JointLimits",
                        joint_name,
                        f"Range is {span:.6f} rad (> pi). For MG90S servos, cap to 180 deg.",
                    )
                )

    return findings


def print_report(findings: Iterable[Finding]) -> Tuple[int, int]:
    failures = 0
    warnings = 0

    for item in findings:
        if item.severity == "FAIL":
            failures += 1
        else:
            warnings += 1
        print(f"[{item.severity}] {item.category} :: {item.target} -> {item.message}")

    return failures, warnings


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check URDF quality for RL locomotion in MuJoCo/MJLAB."
    )
    parser.add_argument("urdf", type=Path, help="Path to the URDF file.")
    parser.add_argument(
        "--mesh-warn-triangles",
        type=int,
        default=5000,
        help="Warn when collision STL triangle count >= this value (default: 5000).",
    )
    parser.add_argument(
        "--mesh-fail-triangles",
        type=int,
        default=50000,
        help="Fail when collision STL triangle count >= this value (default: 50000).",
    )
    parser.add_argument(
        "--no-servo-180-check",
        action="store_true",
        help="Disable Sesame/MG90S 180-degree joint span warning check.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.mesh_warn_triangles > args.mesh_fail_triangles:
        print("Error: --mesh-warn-triangles cannot be greater than --mesh-fail-triangles.")
        return 2

    if not args.urdf.exists():
        print(f"Error: URDF not found: {args.urdf}")
        return 2

    try:
        root = ET.parse(args.urdf).getroot()
    except ET.ParseError as exc:
        print(f"Error: Failed to parse URDF XML: {exc}")
        return 2

    findings: List[Finding] = []
    findings.extend(
        check_collision_vs_visual(
            args.urdf,
            root,
            mesh_warn_triangles=args.mesh_warn_triangles,
            mesh_fail_triangles=args.mesh_fail_triangles,
        )
    )
    findings.extend(check_inertial_mass(root))
    findings.extend(
        check_joint_limits(root, enforce_servo_180=not args.no_servo_180_check)
    )

    failures, warnings = print_report(findings)

    print("\nSummary")
    print(f"  FAIL: {failures}")
    print(f"  WARN: {warnings}")
    print(f"  Total findings: {failures + warnings}")

    if failures > 0:
        print("Result: FAIL (hard blockers found)")
        return 1

    print("Result: PASS (no hard blockers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))