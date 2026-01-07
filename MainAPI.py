import os
import io
import json
import re
from typing import Dict, List, Any, Optional

import pandas as pd
import streamlit as st
try:
	from openpyxl import Workbook
	_HAS_OPENPYXL = True
except Exception:
	Workbook = None
	_HAS_OPENPYXL = False

try:
	import jsonschema
	_HAS_JSONSCHEMA = True
except Exception:
	_HAS_JSONSCHEMA = False

try:
	from google import genai
	from google.genai.errors import APIError
	_HAS_GENAI = True
except Exception:
	genai = None
	APIError = Exception
	_HAS_GENAI = False


APP_TITLE = "OpenAPI -> Test Case Generator (Gemini)"


def parse_openapi(openapi: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
	"""Return mapping module_name -> {operationId -> operation_obj}
	Modules are derived from tags if present, otherwise grouped by first path segment.
	"""
	modules: Dict[str, Dict[str, Any]] = {}

	paths = openapi.get("paths", {})

	for path, methods in paths.items():
		for method, op in methods.items():
			if method.lower() not in ("get", "post", "put", "delete", "patch", "options", "head"):
				continue

			tags = op.get("tags") or []
			module = tags[0] if tags else None
			if not module:
				# fallback: use first path segment after leading /
				segs = [s for s in path.split("/") if s]
				module = segs[0] if segs else "default"

			op_id = op.get("operationId") or f"{method.upper()} {path}"
			display = f"{method.upper()} {path}"

			modules.setdefault(module, {})[op_id] = {
				"method": method.upper(),
				"path": path,
				"operation": op,
				"display": display,
			}

	return modules


def generate_test_cases_from_operation(module_name: str, op_meta: Dict[str, Any], max_cases: int = 5, client: Optional[Any] = None) -> pd.DataFrame:
	"""Generate test cases using Gemini model if available, otherwise create simple deterministic cases from OpenAPI schema.
	Output DataFrame columns must match the required Excel columns.
	"""
	cols = [
		"TestCase_ID",
		"TestCase_Description",
		"Pre-condition",
		"Steps Summary",
		"Payload_test_data",
		"Expected_result",
		"Actual Result",
		"Status",
		"Complexity",
		"Priority",
		"Severity",
		"Comments",
	]

	rows: List[Dict[str, Any]] = []

	op = op_meta.get("operation", {})
	summary = op.get("summary") or op.get("description") or op_meta.get("display")

	def schema_to_example(schema: Dict[str, Any]) -> Any:
		"""Simple recursive example generator from a JSON Schema (OpenAPI subset).
		This is conservative: respects types and required fields, uses examples when provided.
		"""
		if not isinstance(schema, dict):
			return None
		if 'example' in schema:
			return schema['example']
		t = schema.get('type')
		if t == 'object' or ('properties' in schema):
			out = {}
			props = schema.get('properties', {})
			required = set(schema.get('required', []))
			for k, subs in props.items():
				ex = schema_to_example(subs)
				if ex is None and k in required:
					# provide simple defaults for required
					stype = subs.get('type')
					if stype == 'string':
						ex = subs.get('example') or f"example_{k}"
					elif stype in ('integer', 'number'):
						ex = subs.get('example') or 1
					elif stype == 'boolean':
						ex = subs.get('example') or False
					else:
						ex = subs.get('example') or None
				if ex is not None:
					out[k] = ex
			return out
		if t == 'array' or 'items' in schema:
			items = schema.get('items', {})
			return [schema_to_example(items)]
		if t == 'string':
			fmt = schema.get('format')
			if fmt == 'date':
				return schema.get('example') or '2020-01-01'
			return schema.get('example') or 'example'
		if t in ('integer', 'number'):
			return schema.get('example') or 1
		if t == 'boolean':
			return schema.get('example') or False
		return schema.get('example') or None

	# Try model generation (if client available)
	if client is not None:
		# Build a stronger, role-clarified prompt and include schema when available
		req_body = op.get('requestBody') or {}
		content = req_body.get('content', {})
		request_schema = None
		for ct, info in content.items():
			s = info.get('schema') or {}
			# prefer explicit top-level schema object
			if s:
				request_schema = s
				break

		schema_json = json.dumps(request_schema, ensure_ascii=False) if request_schema else '{}'

		prompt = (
			"You are an expert QA engineer and must act strictly within the provided OpenAPI schema. "
			f"Do NOT add fields not defined by the schema.\nModule: {module_name}\nOperation: {op_meta.get('method')} {op_meta.get('path')}\n"
			f"Description: {summary}\n"
			f"Request schema (JSON): {schema_json}\n"
			f"Produce up to {max_cases} test cases as a JSON array. Each test case must be an object with these keys: TestCase_Description, Pre-condition, Steps Summary, Payload_test_data, Expected_result, Complexity, Priority, Severity.\n"
			"For Payload_test_data, produce realistic examples that validate both positive and edge cases (boundary values, missing required fields, invalid types). Do NOT include fields outside the schema. Return only valid JSON."
		)

		# Allow user to pick a model if available (list models)
		try:
			available_models = []
			for m in client.models.list().models:
				mn = getattr(m, 'name', None) or getattr(m, 'model', None) or str(m)
				if mn:
					available_models.append(mn)
		except Exception:
			available_models = []

		# preferred model candidates
		preferred = ["gemini-2.5-flash", "gemini-2.5", "gemini-2.1", "gemini-1.0", "chat-bison"]
		candidates = [m for m in preferred if m in available_models] or available_models or preferred

		# pick the first candidate (or let caller pass one later)
		chosen_model = candidates[0] if candidates else None

		# If running inside Streamlit we can provide the model choice UI; but this function may be called headless.
		try:
			# streamlit available in module scope; show selection if in interactive context
			if st._is_running_with_streamlit and available_models:
				chosen_model = st.selectbox("Model for generation", options=candidates, index=0)
		except Exception:
			pass

		if chosen_model:
			try:
				resp = client.models.generate_content(model=chosen_model, contents=prompt)
				text = resp.text
				# attempt to parse JSON from response
				j = None
				try:
					j = json.loads(text)
				except Exception:
					# try to extract JSON block
					m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
					if m:
						try:
							j = json.loads(m.group(0))
						except Exception:
							j = None

				if isinstance(j, list):
					idx = 1
					for item in j[:max_cases]:
						row = {c: "" for c in cols}
						row["TestCase_ID"] = f"TC_{idx}"
						row["TestCase_Description"] = item.get("TestCase_Description") or item.get("description") or item.get("title") or f"Check {summary}"
						row["Pre-condition"] = item.get("Pre-condition") or item.get("precondition") or ""
						row["Steps Summary"] = item.get("Steps Summary") or item.get("steps") or ""
						row["Payload_test_data"] = json.dumps(item.get("Payload_test_data") or item.get("payload") or {}, ensure_ascii=False)
						row["Expected_result"] = item.get("Expected_result") or item.get("expected") or ""
						row["Actual Result"] = ""
						row["Status"] = "Not Executed"
						row["Complexity"] = item.get("Complexity") or "Medium"
						row["Priority"] = item.get("Priority") or "Medium"
						row["Severity"] = item.get("Severity") or "Normal"
						row["Comments"] = ""
						rows.append(row)
						idx += 1
					if rows:
						return pd.DataFrame(rows)
			except APIError as e:
				# specific API error responses often include helpful messages; surface them
				st.warning(f"Model API error: {e}")
			except Exception as e:
				st.warning(f"Model generation failed: {e}")
		else:
			st.info("No model selected/available for generation; falling back to deterministic generation.")

	# Fallback deterministic generation from parameters/schema
	req_body = op.get('requestBody') or {}
	content = req_body.get('content', {})
	request_schema = None
	for ct, info in content.items():
		request_schema = info.get('schema') or request_schema

	example = None
	if request_schema:
		example = schema_to_example(request_schema)

	idx = 1
	for i in range(max_cases):
		row = {c: '' for c in cols}
		row['TestCase_ID'] = f"TC_{idx}"
		row['TestCase_Description'] = f"{op_meta.get('method')} {op_meta.get('path')} - case {idx}"
		row['Pre-condition'] = 'Service accessible'
		row['Steps Summary'] = f"Call {op_meta.get('method')} {op_meta.get('path')} and verify response code and body"
		# produce variants: 0 -> valid example, 1 -> missing required, 2 -> invalid type, others -> empty
		if i == 0 and example is not None:
			payload = example
		elif i == 1 and isinstance(example, dict):
			# remove one required field if present
			payload = dict(example)
			# try delete a required field
			reqs = request_schema.get('required', []) if isinstance(request_schema, dict) else []
			if reqs:
				payload.pop(reqs[0], None)
		elif i == 2 and isinstance(example, dict):
			# invalid type: set first property to wrong type
			payload = dict(example)
			props = list(payload.keys())
			if props:
				payload[props[0]] = 'INVALID_TYPE'
		else:
			payload = {}

		# validate payload where possible
		if _HAS_JSONSCHEMA and request_schema is not None:
			try:
				jsonschema.validate(payload, request_schema)
				valid_note = ''
			except Exception as e:
				valid_note = f' (schema validation error: {e})'
		else:
			valid_note = ''

		row['Payload_test_data'] = json.dumps(payload, ensure_ascii=False)
		row['Expected_result'] = 'Response conforms to schema and returns appropriate status code' + valid_note
		row['Actual Result'] = ''
		row['Status'] = 'Not Executed'
		row['Complexity'] = 'Medium'
		row['Priority'] = 'Medium'
		row['Severity'] = 'Normal'
		row['Comments'] = ''
		rows.append(row)
		idx += 1

	return pd.DataFrame(rows)


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
	bio = io.BytesIO()
	with pd.ExcelWriter(bio, engine="openpyxl") as writer:
		df.to_excel(writer, sheet_name=sheet_name, index=False)
	return bio.getvalue()


def main():
	st.set_page_config(page_title=APP_TITLE, layout="wide")
	st.title(APP_TITLE)

	# Check runtime dependencies gracefully
	if not _HAS_OPENPYXL:
		st.error("Required package 'openpyxl' is not installed in the environment. Add 'openpyxl' to requirements.txt and redeploy.")
		return

	st.markdown("Upload an OpenAPI JSON file, choose a module (tag) and operation, then generate test cases and download as Excel.")

	# --- Gemini API Key input (secure) ---
	if 'gemini_key' not in st.session_state:
		st.session_state['gemini_key'] = ''
	if 'genai_client' not in st.session_state:
		st.session_state['genai_client'] = None

	with st.expander('Gemini API Key (required for model generation)'):
		key = st.text_input('Enter your Gemini API key', type='password', value=st.session_state.get('gemini_key', ''))
		col1, col2 = st.columns([1, 3])
		with col1:
			if st.button('Set Key'):
				st.session_state['gemini_key'] = key
				# initialize client for session
				if _HAS_GENAI and key:
					os.environ['GEMINI_API_KEY'] = key
					try:
						st.session_state['genai_client'] = genai.Client()
					except Exception as e:
						st.error(f'Failed to initialize Gemini client: {e}')
				else:
					st.warning('google-genai client not installed; model features disabled')
		with col2:
			st.write('The key is stored only for your current Streamlit session and not saved to disk.')

	# Block if key/client missing when trying to use model features
	use_model = False
	if st.session_state.get('genai_client') is not None:
		use_model = True
	elif st.session_state.get('gemini_key'):
		# try to create client lazily
		if _HAS_GENAI:
			try:
				os.environ['GEMINI_API_KEY'] = st.session_state['gemini_key']
				st.session_state['genai_client'] = genai.Client()
				use_model = True
			except Exception:
				use_model = False
		else:
			use_model = False


	uploaded = st.file_uploader("OpenAPI JSON file", type=["json"], accept_multiple_files=False)

	if uploaded is None:
		st.info("Please upload an OpenAPI JSON file to begin.")
		return

	try:
		openapi = json.load(uploaded)
	except Exception as e:
		st.error(f"Failed to parse JSON: {e}")
		return

	# Minimal validation
	if not isinstance(openapi, dict) or "paths" not in openapi:
		st.error("Uploaded file does not look like a valid OpenAPI JSON (missing 'paths').")
		return

	modules = parse_openapi(openapi)
	if not modules:
		st.error("No operations found in the OpenAPI file.")
		return

	module_names = sorted(modules.keys())
	col1, col2 = st.columns([1, 2])

	with col1:
		module_sel = st.selectbox("Module (derived from tags or path)", module_names)
		ops = modules.get(module_sel, {})
		op_items = [f"{k} | {v.get('display')}" for k, v in ops.items()]
		op_choice = st.selectbox("Operation", op_items)

	with col2:
		st.write("Operation details")
		selected_key = op_choice.split(" | ", 1)[0]
		op_meta = ops.get(selected_key)
		st.json({
			"method": op_meta.get("method"),
			"path": op_meta.get("path"),
			"summary": op_meta.get("operation", {}).get("summary"),
			"description": op_meta.get("operation", {}).get("description"),
			"operationId": selected_key,
		})

	st.markdown("---")
	num_cases = st.number_input("Number of test cases to generate", min_value=1, max_value=50, value=5)
	if st.button("Generate Test Cases"):
		with st.spinner("Generating test cases..."):
			# Provide session client if available
			session_client = st.session_state.get('genai_client')
			if session_client is None and _HAS_GENAI and st.session_state.get('gemini_key'):
				# probably invalid key or failed init
				st.error('Gemini API key present but client initialization failed. Check your key or click Set Key again.')
				df = generate_test_cases_from_operation(module_sel, op_meta, max_cases=int(num_cases), client=None)
			else:
				df = generate_test_cases_from_operation(module_sel, op_meta, max_cases=int(num_cases), client=session_client)
			if df is None or df.empty:
				st.warning("No test cases generated.")
			else:
				st.success(f"Generated {len(df)} test cases")
				st.dataframe(df)

				excel_bytes = df_to_excel_bytes(df, sheet_name=(module_sel[:31] or "Sheet1"))
				st.download_button("Download Excel", data=excel_bytes, file_name=f"{module_sel}_testcases.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
	main()
