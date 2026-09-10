"""
controller/agentic_controller.py
-----------------------------------
SatQueryController: the agentic orchestrator.

Pipeline for every request:

    input validator -> router/planner (query_parser) -> executor (workflows)
    -> vision-language model + measured numerical analysis -> synthesizer
    -> visual evidence -> answer + observable execution trace

Division of labour
------------------
The fine-tuned vision-language model owns *semantics*: what the scene is, which
land-cover classes are present, how to describe it. This repository owns
*measurement and geometry*: pixel counts, change fractions, region counts,
masks, overlays, co-registration.

The two are never blended. The synthesizer labels which half of every answer
came from where, so a model that is still loading, out of memory, or missing a
token degrades the semantic half only — every number the UI shows is still a
real measurement of the uploaded pixels.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from api.hf_model import HFModelRunner, ModelAnswer, get_runner
from controller.execution_trace import ExecutionTrace
from controller.input_validator import validate
from controller.query_parser import TaskPlan, parse
from controller import synthesizer
from models import captioning, change_analysis, grounding, optical_sar_fusion, vqa
from models.base_vlm import RemoteSensingVLM
from utils import evidence as evidence_utils
from utils import numerical
from utils.image_io import LoadedImage, align_pair, load_image, to_display_rgb
from utils.spectral_indices import BAND_PRESETS, BandRoles, LAND_COVER_CLASSES
from utils.visualization import draw_bboxes, overlay_binary_mask, overlay_class_map


@dataclass
class ExecutionResult:
    """Everything a client needs to render one completed request."""

    success: bool
    task: str
    scenario: str
    answer: str
    confidence: Optional[float]
    confidence_kind: str = "none"      # "generation" | "heuristic" | "none"
    visual_evidence_path: Optional[str] = None
    evidence_image: Optional[np.ndarray] = None
    measurements: List[str] = field(default_factory=list)
    model_text: Optional[str] = None
    model_labels: List[str] = field(default_factory=list)
    model_source: str = "unknown"
    model_ready: bool = False
    trace_text: str = ""
    trace_steps: List[Dict[str, Any]] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "vqa": {"model": "hf:vision-language", "local": "models.vqa.answer"},
    "caption": {"model": "hf:vision-language", "local": "models.captioning.caption"},
    "grounding": {"model": "", "local": "models.grounding.ground"},
    "fusion_analysis": {"model": "hf:vision-language (per modality)",
                        "local": "models.optical_sar_fusion.fuse"},
    "change_description": {"model": "hf:vision-language (per date)",
                           "local": "models.change_analysis.describe_change"},
    "change_vqa": {"model": "hf:vision-language (per date)",
                   "local": "models.change_analysis.change_vqa"},
}

_WORKFLOW_LABELS = {
    "single": "Single image analysis",
    "bi_temporal": "Bi-temporal change detection",
    "cross_modal": "Optical + SAR cross-modal analysis",
}


class SatQueryController:
    def __init__(
        self,
        vlm: Optional[RemoteSensingVLM] = None,
        reports_dir: str = "reports",
        caption_checkpoint: Optional[str] = None,
        runner: Optional[HFModelRunner] = None,
        use_model: bool = True,
    ):
        self.vlm = vlm or RemoteSensingVLM()
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        self.use_model = use_model
        self.runner = runner or get_runner()

        self.caption_model = None
        self.caption_model_error = None
        checkpoint = caption_checkpoint or os.environ.get("SATQUERY_CAPTION_CHECKPOINT")
        if checkpoint:
            try:
                self.caption_model = captioning.CaptionModel.load(checkpoint)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.caption_model_error = (
                    f"Caption checkpoint '{checkpoint}' could not be loaded: {exc}"
                )

    # ------------------------------------------------------------------ #
    def model_status(self) -> Dict[str, Any]:
        try:
            return self.runner.status()
        except Exception as exc:
            return {"state": "error", "message": f"Status check failed: {exc}",
                    "progress": [], "warnings": []}

    def run(
        self,
        image_paths: List[str],
        query: str,
        declared_pair_type: Optional[str] = None,
        band_preset: Optional[str] = None,
    ) -> ExecutionResult:
        trace = ExecutionTrace()
        started = time.time()
        band_roles = BAND_PRESETS.get(band_preset) if band_preset else None

        # --- 1. load ------------------------------------------------------ #
        try:
            loaded = [load_image(path) for path in image_paths]
        except Exception as exc:
            trace.fail("Image validation", str(exc))
            return self._failure(
                trace, "invalid", "invalid",
                "Unsupported or invalid image. Please upload a valid satellite image.",
                str(exc),
            )

        trace.ok(
            "Image validation",
            "; ".join(
                f"{os.path.basename(img.path)}: {img.width}x{img.height}, "
                f"{img.bands} band(s), modality={img.modality_guess}"
                for img in loaded
            ),
            timed=True,
        )

        # --- 2. validate --------------------------------------------------- #
        validation = validate(loaded, declared_pair_type=declared_pair_type)
        if not validation.ok:
            trace.fail("Input compatibility check", " ".join(validation.messages))
            return self._failure(
                trace, "invalid", validation.scenario,
                self._validation_message(loaded, validation.messages),
                "Input validation failed.",
            )
        scenario = validation.scenario
        trace.ok("Input compatibility check", " ".join(validation.messages[-2:]), timed=True)

        if band_roles:
            trace.info("Band roles", f"{band_preset} -> {band_roles.as_dict()}")
        else:
            trace.info(
                "Band roles",
                "No preset supplied; spectral indices fall back to RGB colour proxies.",
            )

        # --- 3. route ------------------------------------------------------ #
        plan: TaskPlan = parse(query, scenario)
        if plan.task == "invalid":
            trace.fail("Query classification", plan.rationale)
            return self._failure(
                trace, "invalid", scenario,
                "The query could not be mapped to a supported task for these inputs.",
                "Unroutable query.",
            )
        trace.ok("Query classification", f"task={plan.task} — {plan.rationale}", timed=True)
        trace.ok(
            f"Workflow selected: {_WORKFLOW_LABELS.get(scenario, scenario)}",
            f"registry entry: {MODEL_REGISTRY.get(plan.task, {})}",
        )

        # --- 4. model readiness -------------------------------------------- #
        model_ready = False
        model_note: Optional[str] = None
        if self.use_model:
            status = self.runner.status()
            if status["state"] == "ready":
                model_ready = True
                trace.ok(
                    "Vision-language model ready",
                    f"{status['model_id']} on {status['device']}"
                    + (f" (adapter on {status['base_model_id']})" if status["is_adapter"] else ""),
                    timed=True,
                )
            else:
                self.runner.ensure_loading()
                # Re-read after kicking off the load: a fast failure (no torch,
                # bad token) resolves immediately, and reporting the real reason
                # beats repeating a stale "not loaded yet".
                status = self.runner.status()
                model_note = status["message"]
                trace.warn(
                    f"Vision-language model {status['state']}",
                    f"{model_note} Continuing with local specialists; all measured "
                    f"statistics are unaffected.",
                )
        else:
            model_note = "Model disabled for this run."
            trace.info("Vision-language model skipped", model_note)

        # --- 5. execute ------------------------------------------------------#
        try:
            result, board = self._execute(
                plan, loaded, scenario, trace, band_roles, model_ready, model_note
            )
        except Exception as exc:
            trace.fail("Workflow execution", f"{type(exc).__name__}: {exc}")
            return self._failure(
                trace, plan.task, scenario,
                f"The analysis could not be completed: {exc}", str(exc),
            )

        # --- 6. persist ------------------------------------------------------#
        evidence_path = None
        if board is not None:
            try:
                from PIL import Image

                evidence_path = os.path.join(self.reports_dir, f"evidence_{int(started)}.png")
                Image.fromarray(board).save(evidence_path)
                trace.ok("Visual evidence generated", os.path.basename(evidence_path))
            except OSError as exc:
                trace.warn("Visual evidence generated", f"Could not write to disk: {exc}")
        else:
            trace.warn("Visual evidence generated", "No panels available for this workflow.")

        trace.ok("Final response generated")

        answer_text = result.get("answer") or ""
        report_path = os.path.join(self.reports_dir, f"report_{int(started)}.json")
        try:
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "task": plan.task,
                    "scenario": scenario,
                    "query": query,
                    "answer": answer_text,
                    "confidence": result.get("confidence"),
                    "measurements": result.get("measurements", []),
                    "model_source": result.get("model_source"),
                    "model_ready": model_ready,
                    "trace": trace.as_dicts(),
                    "elapsed_seconds": time.time() - started,
                }, handle, indent=2, default=str)
        except OSError:
            pass

        return ExecutionResult(
            success=True,
            task=plan.task,
            scenario=scenario,
            answer=answer_text,
            confidence=result.get("confidence"),
            confidence_kind=result.get("confidence_kind", "none"),
            visual_evidence_path=evidence_path,
            evidence_image=board,
            measurements=result.get("measurements", []),
            model_text=result.get("model_text"),
            model_labels=result.get("model_labels", []),
            model_source=result.get("model_source", "local specialist"),
            model_ready=model_ready,
            trace_text=trace.render(),
            trace_steps=trace.as_dicts(),
            audit_trail=trace.as_list(),
            raw={k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
        )

    # ------------------------------------------------------------------ #
    def _execute(self, plan, loaded, scenario, trace, band_roles, model_ready, model_note):
        if scenario == "single":
            return self._run_single(plan, loaded[0], trace, band_roles, model_ready, model_note)
        if scenario == "bi_temporal":
            return self._run_change(plan, loaded, trace, band_roles, model_ready, model_note)
        if scenario == "cross_modal":
            return self._run_cross_modal(plan, loaded, trace, band_roles, model_ready, model_note)
        raise RuntimeError(f"No execution path for scenario={scenario}, task={plan.task}")

    # -- A. Single image -------------------------------------------------- #
    def _run_single(self, plan, image, trace, band_roles, model_ready, model_note):
        local_evidence = self.vlm.encode(image.array, band_roles=band_roles)
        trace.ok("Land-cover analysis",
                 local_evidence.notes[0] if local_evidence.notes else None, timed=True)

        stats = numerical.image_statistics(image)
        measurements = list(stats.as_lines())
        rgb = to_display_rgb(image)
        panels: List[Tuple[str, np.ndarray]] = [("Original image", rgb)]

        answer_obj = ModelAnswer.failure(model_note or "Model not queried")
        if model_ready:
            answer_obj = self.runner.analyze(image.array, plan.query)
            if answer_obj.ok:
                trace.ok("Vision-language inference", answer_obj.summary_line(), timed=True)
            else:
                trace.warn("Vision-language inference", answer_obj.error or "no result")

        # The local specialist always runs: it supplies the overlay and the
        # measurements, and answers if the model could not.
        if plan.task == "grounding":
            target = plan.params.get("target_class") or "built_up"
            local_result = grounding.ground(local_evidence, target)
            overlay = overlay_binary_mask(rgb, local_result["mask"])
            if local_result.get("bboxes"):
                overlay = draw_bboxes(overlay, local_result["bboxes"],
                                      labels=[target] * len(local_result["bboxes"]))
            panels.append((f"Located: {target.replace('_', ' ')}", overlay))
            region = numerical.change_statistics(
                local_result["mask"], pixel_size_m=image.pixel_size_m,
                has_geo_metadata=image.has_geo_metadata)
            measurements.append(
                f"{target.replace('_', ' ')} pixels: {region.changed_pixels:,} "
                f"({region.changed_percent:.2f}% of image)")
            if region.region_count:
                measurements.append(f"Distinct regions found: {region.region_count}")
            trace.ok("Region grounding", f"target={target}", timed=True)
        elif plan.task == "caption":
            # Which captioner answered is audit-relevant: a dataset checkpoint
            # and the spectral fallback produce very different quality.
            if self.caption_model is not None:
                trace.ok("Caption backend", "dataset-nearest-neighbor checkpoint")
            elif self.caption_model_error:
                trace.warn("Caption backend",
                           f"{self.caption_model_error}; using the heuristic fallback")
            else:
                trace.info("Caption backend",
                           "heuristic spectral fallback (no checkpoint configured)")
            local_result = captioning.caption(local_evidence, image=image.array,
                                              model=self.caption_model)
            panels.append(("Land-cover classification",
                           overlay_class_map(rgb, local_evidence.class_map,
                                             self.vlm.class_names(), alpha=0.45)))
        else:
            local_result = vqa.answer(local_evidence, plan.query)
            panels.append(("Land-cover classification",
                           overlay_class_map(rgb, local_evidence.class_map,
                                             self.vlm.class_names(), alpha=0.45)))

        for row in numerical.class_area_table(local_evidence.proportions,
                                              stats.total_pixels, image.pixel_size_m):
            if row["percent"] >= 0.5:
                measurements.append(
                    f"{row['class'].replace('_', ' ')}: {row['percent']:.1f}% "
                    f"({row['pixels']:,} px)")
        trace.ok("Numerical analysis", f"{len(measurements)} measured values", timed=True)

        local_text = local_result.get("answer") or local_result.get("caption") or ""
        board = evidence_utils.compose_board(
            panels, legend=evidence_utils.landcover_legend(LAND_COVER_CLASSES))

        model_lines = synthesizer.describe_model_output(answer_obj)
        notes: List[str] = []
        if answer_obj.ok and answer_obj.text:
            headline = answer_obj.text
            confidence, confidence_kind = answer_obj.generation_confidence, "generation"
            model_source = "fine-tuned vision-language model"
            if local_text:
                notes.append(f"Local specialist cross-check: {local_text}")
        else:
            headline = local_text or "No answer could be produced for this image."
            confidence, confidence_kind = local_result.get("confidence"), "heuristic"
            model_source = "local specialist"
            notes.append(synthesizer.model_unavailable_notice(answer_obj.error))

        local_result.update({
            "answer": synthesizer.compose(
                headline=headline, measurement_lines=measurements,
                model_lines=model_lines, notes=notes,
                provenance=synthesizer.provenance_line(
                    model_used=answer_obj.ok, local_used=True, measured=True)),
            "confidence": confidence,
            "confidence_kind": confidence_kind,
            "measurements": measurements,
            "model_text": answer_obj.text,
            "model_labels": answer_obj.labels,
            "model_source": model_source,
        })
        return local_result, board

    # -- B. Bi-temporal change detection ----------------------------------- #
    def _run_change(self, plan, loaded, trace, band_roles, model_ready, model_note):
        before, after = loaded[0], loaded[1]
        array_before, array_after, align_note = align_pair(before, after)
        trace.ok("T1/T2 preprocessing",
                 align_note or "Both dates already share a pixel grid.", timed=True)

        evidence_before = self.vlm.encode(array_before, band_roles=band_roles)
        evidence_after = self.vlm.encode(array_after, band_roles=band_roles)

        change = change_analysis.compute_change_map(array_before, array_after)
        change["water_reliable"] = any(
            "true NDWI" in note or "true MNDWI" in note
            for note in evidence_before.notes + evidence_after.notes)
        trace.ok("Change map generated",
                 "pixel differencing on chroma-normalised bands, Otsu threshold", timed=True)

        pixel_size = before.pixel_size_m or after.pixel_size_m
        stats = numerical.change_statistics(
            change["mask"], pixel_size_m=pixel_size,
            has_geo_metadata=before.has_geo_metadata and after.has_geo_metadata)
        measurements = list(stats.as_lines())
        trace.ok("Changed area calculated",
                 f"{stats.changed_pixels:,} px = {stats.changed_percent:.2f}% "
                 f"across {stats.region_count} region(s)", timed=True)

        # The model describes each date. It is never asked to quantify.
        answers: List[ModelAnswer] = []
        label_before = label_after = None
        if model_ready:
            answers = self.runner.analyze_many([
                (array_before, plan.query),
                (array_after, plan.query),
            ])
            label_before = answers[0].text if answers[0].ok else None
            label_after = answers[1].text if answers[1].ok else None
            statuses = " | ".join(a.summary_line() for a in answers)
            if label_before or label_after:
                trace.ok("Vision-language inference (T1, T2)", statuses, timed=True)
            else:
                trace.warn("Vision-language inference (T1, T2)", statuses)

        local_result = (
            change_analysis.change_vqa(evidence_before, evidence_after, change,
                                       plan.query, plan.params.get("target_class"))
            if plan.task == "change_vqa"
            else change_analysis.describe_change(evidence_before, evidence_after, change))

        headline = synthesizer.change_narrative(
            changed_percent=stats.changed_percent,
            location=stats.centroid_description,
            class_shift=self._class_shift_sentence(
                evidence_before.proportions, evidence_after.proportions,
                change["water_reliable"]),
            before_label=_first_sentence(label_before),
            after_label=_first_sentence(label_after))

        rgb_before = evidence_utils.to_uint8_rgb(array_before)
        rgb_after = evidence_utils.to_uint8_rgb(array_after)
        board = evidence_utils.compose_board([
            ("T1 — earlier date", rgb_before),
            ("T2 — later date", rgb_after),
            ("Difference magnitude",
             evidence_utils.heatmap(numerical.difference_map(array_before, array_after))),
            ("Change mask on T2",
             overlay_binary_mask(rgb_after, change["mask"], color=(255, 46, 196))),
        ], legend=[("changed pixels", (255, 46, 196))])

        model_lines: List[str] = []
        for index, answer_obj in enumerate(answers):
            for line in synthesizer.describe_model_output(answer_obj):
                model_lines.append(f"T{index + 1} — {line}")

        notes: List[str] = []
        if not model_ready:
            notes.append(synthesizer.model_unavailable_notice(model_note))
            notes.append("All change measurements above were computed directly from "
                         "the two images and are unaffected.")
        if not change["water_reliable"]:
            notes.append("Water change is not reported: without a NIR/SWIR band, RGB "
                         "alone cannot separate a real water change from a lighting shift.")
        if local_result.get("answer"):
            notes.append(f"Local specialist cross-check: {local_result['answer']}")

        confidences = [a.generation_confidence for a in answers
                       if a.generation_confidence is not None]
        if confidences:
            confidence, confidence_kind = sum(confidences) / len(confidences), "generation"
        else:
            confidence, confidence_kind = local_result.get("confidence"), "heuristic"

        local_result.update({
            "answer": synthesizer.compose(
                headline=headline, measurement_lines=measurements,
                model_lines=model_lines, notes=notes,
                provenance=synthesizer.provenance_line(
                    model_used=bool(label_before or label_after),
                    local_used=True, measured=True)),
            "confidence": confidence,
            "confidence_kind": confidence_kind,
            "measurements": measurements,
            "model_text": label_after or label_before,
            "model_labels": sorted({l for a in answers for l in a.labels}),
            "model_source": ("fine-tuned vision-language model + measured statistics"
                             if (label_before or label_after)
                             else "local specialist + measured statistics"),
            "changed_pixels": stats.changed_pixels,
            "changed_percent": stats.changed_percent,
            "region_count": stats.region_count,
        })
        return local_result, board

    # -- C. Optical + SAR --------------------------------------------------- #
    def _run_cross_modal(self, plan, loaded, trace, band_roles, model_ready, model_note):
        optical = next((img for img in loaded if img.modality_guess != "sar"), loaded[0])
        sar = next((img for img in loaded if img is not optical), loaded[1])
        trace.ok("Modality assignment",
                 f"optical='{os.path.basename(optical.path)}', "
                 f"sar='{os.path.basename(sar.path)}'", timed=True)

        optical_array, sar_array, align_note = align_pair(optical, sar)
        if align_note:
            trace.ok("Cross-modal co-registration", align_note)

        optical_evidence = self.vlm.encode(optical_array, band_roles=band_roles)
        fusion = optical_sar_fusion.fuse(optical_evidence, sar_array,
                                         target_class=plan.params.get("target_class"))
        trace.ok("Optical/SAR fusion",
                 "spectral classes refined by SAR backscatter and texture", timed=True)

        optical_answer = ModelAnswer.failure(model_note or "Model not queried")
        sar_answer = ModelAnswer.failure(model_note or "Model not queried")
        if model_ready:
            optical_answer, sar_answer = self.runner.analyze_many([
                (optical_array, plan.query),
                (sar_array, plan.query),
            ])
            trace.ok("Vision-language inference (optical, SAR)",
                     f"optical: {optical_answer.summary_line()} | "
                     f"SAR: {sar_answer.summary_line()}", timed=True)

        total_pixels = int(optical_evidence.class_map.size)
        pixel_size = optical.pixel_size_m
        measurements = [
            f"Optical grid: {optical_array.shape[1]} x {optical_array.shape[0]} px",
            f"SAR grid: {sar_array.shape[1]} x {sar_array.shape[0]} px",
            f"Built-up confirmed by both sensors: "
            f"{fusion['refined_built_up_fraction'] * 100:.2f}% "
            f"({int(fusion['refined_built_up_fraction'] * total_pixels):,} px)",
            f"Water confirmed by both sensors: "
            f"{fusion['refined_water_fraction'] * 100:.2f}% "
            f"({int(fusion['refined_water_fraction'] * total_pixels):,} px)",
            f"Optical-only built-up estimate: "
            f"{fusion['optical_only_built_up_fraction'] * 100:.2f}%",
            f"Optical-only water estimate: {fusion['optical_only_water_fraction'] * 100:.2f}%",
        ]
        if pixel_size:
            area = fusion["refined_built_up_fraction"] * total_pixels * pixel_size ** 2
            measurements.append(f"Built-up ground area: {area / 10_000:.2f} ha")
        else:
            measurements.append("Physical area could not be calculated because spatial "
                                "resolution/CRS metadata was unavailable.")
        trace.ok("Numerical analysis", f"{len(measurements)} measured values", timed=True)

        rgb_optical = evidence_utils.to_uint8_rgb(optical_array)
        rgb_sar = evidence_utils.to_uint8_rgb(sar_array)
        combined = fusion["refined_built_up_mask"] | fusion["refined_water_mask"]
        board = evidence_utils.compose_board([
            ("Optical", rgb_optical),
            ("SAR", rgb_sar),
            ("SAR structure (edges)", evidence_utils.heatmap(fusion["sar_edge_map"])),
            ("Fused: built-up + water",
             overlay_binary_mask(rgb_optical, combined, color=(255, 165, 0))),
        ], legend=[("confirmed built-up / water", (255, 165, 0))])

        sections: List[str] = []
        if optical_answer.ok and optical_answer.text:
            sections.append(f"OPTICAL INTERPRETATION\n  {optical_answer.text}")
        else:
            dominant = sorted(optical_evidence.proportions.items(),
                              key=lambda kv: kv[1], reverse=True)[:2]
            sections.append(
                "OPTICAL INTERPRETATION\n  Spectral classification (local specialist): "
                + ", ".join(f"{n.replace('_', ' ')} {f * 100:.0f}%" for n, f in dominant) + ".")

        if sar_answer.ok and sar_answer.text:
            sections.append(f"SAR INTERPRETATION\n  {sar_answer.text}")
        else:
            gray = sar_array[..., 0]
            sections.append(
                f"SAR INTERPRETATION\n  Backscatter analysis (local specialist): "
                f"{float((gray > 0.6).mean()) * 100:.1f}% of the scene returns strongly "
                f"(rough surfaces such as buildings and vegetation) and "
                f"{float((gray < 0.2).mean()) * 100:.1f}% returns weakly (smooth surfaces "
                f"such as open water or tarmac).")

        sections.append(f"FUSED INTERPRETATION\n  {fusion['answer']}")

        notes: List[str] = []
        if model_ready:
            notes.append("The vision-language model analyses one image at a time, so it "
                         "was run separately on each modality. The fused result above is "
                         "computed by SatQuery's cross-modal specialist, not by the model.")
        else:
            notes.append(synthesizer.model_unavailable_notice(model_note))
            notes.append("Both interpretations above therefore come from local specialists. "
                         "The measured statistics are unaffected.")

        model_lines: List[str] = []
        for label, answer_obj in (("Optical", optical_answer), ("SAR", sar_answer)):
            for line in synthesizer.describe_model_output(answer_obj):
                model_lines.append(f"{label} — {line}")

        confidences = [a.generation_confidence for a in (optical_answer, sar_answer)
                       if a.generation_confidence is not None]
        if confidences:
            confidence, confidence_kind = sum(confidences) / len(confidences), "generation"
        else:
            confidence, confidence_kind = fusion.get("confidence"), "heuristic"

        fusion.update({
            "answer": synthesizer.compose(
                headline="\n\n".join(sections), measurement_lines=measurements,
                model_lines=model_lines, notes=notes,
                provenance=synthesizer.provenance_line(
                    model_used=optical_answer.ok or sar_answer.ok,
                    local_used=True, measured=True)),
            "confidence": confidence,
            "confidence_kind": confidence_kind,
            "measurements": measurements,
            "model_text": optical_answer.text,
            "model_labels": sorted(set(optical_answer.labels) | set(sar_answer.labels)),
            "model_source": ("fine-tuned model (per modality) + local fusion"
                             if (optical_answer.ok or sar_answer.ok) else "local specialists"),
        })
        return fusion, board

    # ------------------------------------------------------------------ #
    @staticmethod
    def _class_shift_sentence(before, after, water_reliable):
        deltas = {n: after.get(n, 0.0) - before.get(n, 0.0) for n in LAND_COVER_CLASSES}
        if not water_reliable:
            deltas["water"] = 0.0
        ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top = [(n, d) for n, d in ranked if abs(d) > 0.02][:2]
        if not top:
            return None
        return "Measured land-cover shift: " + "; ".join(
            f"{n.replace('_', ' ')} {'increased' if d > 0 else 'decreased'} by about "
            f"{abs(d) * 100:.1f} percentage points" for n, d in top) + "."

    @staticmethod
    def _validation_message(loaded, messages):
        if len(loaded) == 0:
            return "No image supplied. Please upload a satellite image to analyse."
        if len(loaded) > 2:
            return (f"{len(loaded)} images supplied. SatQuery analyses one image, or a "
                    f"pair (T1/T2, or optical/SAR).")
        return "The supplied images are not compatible.\n" + "\n".join(messages[-2:])

    def _failure(self, trace, task, scenario, message, error):
        return ExecutionResult(
            success=False, task=task, scenario=scenario, answer=message,
            confidence=None, trace_text=trace.render(), trace_steps=trace.as_dicts(),
            audit_trail=trace.as_list(), error=error)


def _first_sentence(text: Optional[str]) -> Optional[str]:
    """Trim a model reply to its first sentence for use inside a summary line."""
    if not text:
        return None
    cleaned = " ".join(text.split())
    for stop in (". ", "! ", "? "):
        if stop in cleaned:
            cleaned = cleaned.split(stop)[0]
            break
    return cleaned[:180]
