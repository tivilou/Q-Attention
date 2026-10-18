"""Bounded private reader: integrity, arbitrary slices, and HTTP isolation."""
import hashlib
import importlib.util
import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

import numpy as np
import pytest

urlopen=build_opener(ProxyHandler({})).open

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("ecpa_case_reader",ROOT/"scripts/serve_ecpa_case_reader.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    report,cases=tmp_path/"report",tmp_path/"cases"
    fingerprint="a"*64
    jobs={}
    for job in module.MODES:
        path=cases/f"jobs/{job}/sample_trace.json"
        values=path.parent/"values/array.npy"
        values.parent.mkdir(parents=True,exist_ok=True)
        array=np.arange(2*3*41*32,dtype=np.float32).reshape(2,3,41,32).astype(np.complex64)*(1+2j)
        np.save(values,array,allow_pickle=False)
        ref=dict(path="values/array.npy",bytes=values.stat().st_size,sha256=module.sha(values))
        rep=dict(id="layer0_amplitudes",stage_id="layer0_scoring",meaning="Actual complex path states",shape=list(array.shape),dtype=str(array.dtype),
            axis_semantics=["head","step","destination","source"],display=dict(visible_fields=[dict(label="Path states")]),
            value_capture=dict(stored_element_count=array.size,artifact_refs=[ref]))
        sample=dict(sample_id="test-0000:final:replay",lane="replay",split="test",checkpoint="final",roles=dict(sentence="a b",tokens=["a","b"]),
            representations=[rep],stages=[dict(stage_id="layer0_scoring",status="observed",input_refs=["external:model_checkpoint"],output_refs=[rep["id"]])])
        trace=dict(schema_version="sample-trace.v2",trace_completeness="complete",experiment=dict(config_sha256=fingerprint,model_identity=job),
            samples=[sample],sample_selection=dict(rule="first fixed sample"))
        write(path,trace)
        jobs[job]=dict(evidence_path=f"jobs/{job}/sample_trace.json",trace_sha256=module.sha(path))
    write(report/"case_study.json",dict(trace_completeness="complete",jobs=jobs))
    write(report/"run_manifest.json",dict(fingerprint=fingerprint,run_id="fixture",data={}))
    write(report/"run_config.json",dict(profile="synthetic_canary"))
    write(report/"REPORT_COMPLETE.json",dict(schema_version="ecpa-report.v1",fingerprint=fingerprint,
        files={p.name:module.sha(p) for p in report.iterdir()}))
    return report,cases,array


def test_arbitrary_matrix_vector_and_complex_components(evidence):
    report,cases,array=evidence
    reader=module.CaseReader(report,cases)
    for component,fn in (("real",np.real),("imag",np.imag),("magnitude",np.abs)):
        values=reader.values("quantum","test-0000:final:replay","layer0_amplitudes",[1,2],row=32,column=16,component=component)
        np.testing.assert_array_equal(values["values"],fn(array[1,2,32:64,16:48]))
        vector=reader.values("quantum","test-0000:final:replay","layer0_amplitudes",[1,2,40],view="vector",component=component)
        np.testing.assert_array_equal(vector["values"],fn(array[1,2,40]))
    assert reader.project("quantum","test-0000:final:replay")["representations"][0]["direction"]=="output"
    for prefix in ([2,0],[-1,0],[True,0],[0],"../",[0,0,0]):
        with pytest.raises(ValueError):
            reader.values("quantum","test-0000:final:replay","layer0_amplitudes",prefix)


def test_changed_numeric_and_trace_are_rejected(evidence):
    report,cases,_=evidence
    reader=module.CaseReader(report,cases)
    values=cases/"jobs/quantum/values/array.npy"
    with values.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError,match="integrity"):
        reader.values("quantum","test-0000:final:replay","layer0_amplitudes",[0,0])
    trace=cases/"jobs/quantum/sample_trace.json"
    trace.write_text(trace.read_text()+" ")
    with pytest.raises(ValueError,match="trace changed"):
        reader.sample("quantum","test-0000:final:replay")


def test_http_no_raw_path_no_cross_origin_and_read_only(evidence):
    report,cases,_=evidence
    assert len(module.CaseReader(report,cases).index()["jobs"])==8
    server=module.ThreadingHTTPServer(("127.0.0.1",0),module.handler(module.CaseReader(report,cases)))
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    root=f"http://127.0.0.1:{server.server_port}"
    try:
        try:
            response=urlopen(root+"/api/index")
        except HTTPError as exc:
            raise AssertionError(exc.read().decode()) from exc
        with response:
            assert response.headers["Cache-Control"]=="no-store"
            assert len(json.load(response)["jobs"])==8
        for path,headers in (("/files/array.npy",{}),("/../REPORT_COMPLETE.json",{}),("/api/index",{"Origin":"https://elsewhere.invalid"}),("/api/index",{"Host":"elsewhere.invalid"})):
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(root+path,headers=headers))
            assert exc.value.code in (403,404)
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(root+"/api/index",data=b"x",method="POST"))
        assert exc.value.code==501
        payload=urlopen(root+"/api/sample?job=quantum&sample=test-0000%3Afinal%3Areplay").read().decode()
        assert str(cases) not in payload and "array.npy" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_report_hash_and_paths_fail_closed(evidence):
    report,cases,_=evidence
    with pytest.raises(ValueError):
        module.inside(cases,"../report/REPORT_COMPLETE.json")
    (report/"run_config.json").write_text('{}')
    with pytest.raises(ValueError,match="report integrity"):
        module.CaseReader(report,cases)
