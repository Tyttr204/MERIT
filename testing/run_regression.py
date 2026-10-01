#!/usr/bin/env python
"""Run explicit regression fixtures against existing main/fusion assets.
No third-party packages are installed and no EAV data are read. No overwrite.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output and args.output.exists():p.error('Output already exists; use a new report path')
    root=Path(__file__).resolve().parent
    os.environ['EAV_DEPLOYMENT_ROOT']=str(args.deployment_root.resolve())
    spec=importlib.util.spec_from_file_location('eav_runner_regression',root/'tests/test_system_test_runner.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    suite=unittest.defaultTestLoader.loadTestsFromModule(mod)
    res=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS' if res.wasSuccessful() else 'FAILED','tests_run':res.testsRun,
        'failures':len(res.failures),'errors':len(res.errors),'skipped':len(res.skipped),
        'failure_details':[{'test':str(t),'traceback':s} for t,s in res.failures+res.errors],
        'skip_details':[{'test':str(t),'reason':s} for t,s in res.skipped],
        'runner_sha256':mod.R.digest(root/'system_test_runner.py'),
        'self_test':mod.R.self_test(),'environment':mod.R.environment(),
        'scope':{'actual_user_F4_AF4B':True,'raw_fixtures':'GENERATED_SYNTHETIC',
                 'emotion_and_quality_heads':'EXPLICIT_TEST_DOUBLES',
                 'real_main_SourceReader_and_Runtime_process_used':True,
                 'FFmpeg_real_transformations':True,'real_EAV_read':False,
                 'PyPREP_DNSMOS_DOVER_E4_emotion2vec_DFER_inference':False,
                 'hardware_used':False,'new_emotion_accuracy_claimed':False}}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
        print('REGRESSION REPORT:',args.output.resolve())
    return 0 if res.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
