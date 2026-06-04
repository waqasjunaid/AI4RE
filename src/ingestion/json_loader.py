"""
json_loader.py
Loads JSON files with structure-aware splitting.

For Swagger/OpenAPI files: splits by endpoint path so each
section = one API endpoint with its methods and parameters.

For other JSON: splits by top-level keys.
"""

import json
import os
from typing import List, Dict, Any
from src.ingestion.base_loader import BaseLoader


class JSONLoader(BaseLoader):

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                # Fall back to treating as plain text
                f.seek(0)
                text = self._clean(f.read())
                return [{"section_id": 1, "section_ref": "json:raw", "text": text}]

        # Detect Swagger / OpenAPI
        if isinstance(data, dict) and ("swagger" in data or "openapi" in data):
            return self._parse_openapi(data)

        # Generic JSON: split by top-level keys
        return self._parse_generic(data)

    def _parse_openapi(self, data: dict) -> List[Dict[str, Any]]:
        sections = []

        # Section 1: API info and metadata
        info = data.get("info", {})
        info_text = "API: {}\nVersion: {}\nDescription: {}\nBase path: {}\nHost: {}".format(
            info.get("title", ""),
            info.get("version", ""),
            info.get("description", ""),
            data.get("basePath", ""),
            data.get("host", ""),
        )
        sections.append({
            "section_id":  1,
            "section_ref": "openapi:info",
            "text":        self._clean(info_text),
        })

        # Section per path/endpoint
        paths = data.get("paths", {})
        for path, methods in paths.items():
            lines = ["Endpoint: {}".format(path)]
            for method, spec in methods.items():
                if not isinstance(spec, dict):
                    continue
                lines.append("  Method: {}".format(method.upper()))
                lines.append("  Summary: {}".format(spec.get("summary", "")))
                lines.append("  Description: {}".format(spec.get("description", "")))
                lines.append("  Tags: {}".format(", ".join(spec.get("tags", []))))

                params = spec.get("parameters", [])
                if params:
                    lines.append("  Parameters:")
                    for p in params:
                        lines.append("    - {} ({}, {}, required={})".format(
                            p.get("name", ""),
                            p.get("in", ""),
                            p.get("type", p.get("schema", {}).get("type", "")),
                            p.get("required", False),
                        ))

                responses = spec.get("responses", {})
                if responses:
                    lines.append("  Responses:")
                    for code, resp in responses.items():
                        desc = resp.get("description", "") if isinstance(resp, dict) else ""
                        lines.append("    - {}: {}".format(code, desc))

            sections.append({
                "section_id":  len(sections) + 1,
                "section_ref": "openapi:path:{}".format(path),
                "text":        self._clean("\n".join(lines)),
            })

        # Definitions/schemas
        defs = data.get("definitions", data.get("components", {}).get("schemas", {}))
        for name, schema in defs.items():
            props = schema.get("properties", {})
            lines = ["Schema: {}".format(name)]
            lines.append("Description: {}".format(schema.get("description", "")))
            for prop, pdef in props.items():
                ptype = pdef.get("type", "")
                pdesc = pdef.get("description", "")
                lines.append("  - {} ({}): {}".format(prop, ptype, pdesc))
            sections.append({
                "section_id":  len(sections) + 1,
                "section_ref": "openapi:schema:{}".format(name),
                "text":        self._clean("\n".join(lines)),
            })

        return [s for s in sections if s["text"]]

    def _parse_generic(self, data: dict) -> List[Dict[str, Any]]:
        sections = []
        if isinstance(data, dict):
            for key, value in data.items():
                text = self._clean(json.dumps(value, indent=2, ensure_ascii=False))
                if text:
                    sections.append({
                        "section_id":  len(sections) + 1,
                        "section_ref": "json:{}".format(key),
                        "text":        text,
                    })
        elif isinstance(data, list):
            for idx, item in enumerate(data, start=1):
                text = self._clean(json.dumps(item, indent=2, ensure_ascii=False))
                if text:
                    sections.append({
                        "section_id":  idx,
                        "section_ref": "json:item:{}".format(idx),
                        "text":        text,
                    })
        return sections
