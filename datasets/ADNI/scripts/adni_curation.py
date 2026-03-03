"""Curation script for the ADNI fMRI benchmark split.

Adapted from https://github.com/mihirneal/ADNI_meta (path-adjusted for this repo).

Inputs:
  - ../metadata/adni_fmri_sessions.json  (public, imaging metadata)
  - ADNIMERGE CSV (private, requires ADNI access — set ADNI_MERGE_CSV env var,
    or place as ../metadata/ADNIMERGE_14May2025.csv; update filename if needed)

Output:
  - ../metadata/adni_fmri_benchmark_split.csv  (consumed by make_adni_targets.py)
"""

import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).parent
_META = _HERE.parent / "metadata"


def generate_benchmark_from_json():
    print("--- MedARC ADNI Benchmark Generation (JSON Source) ---")

    # 1. Load Valid Subjects/Sessions from metadata JSON
    json_file = _META / "adni_fmri_sessions.json"
    if not json_file.exists():
        print(f"ERROR: '{json_file}' not found.")
        return

    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Convert JSON structure to DataFrame
    # New JSON structure: {"subjects": {"002S0295": [{"session_id": "20110602", "TR": 6.02, ...}], ...}}
    scan_records = []
    tr3_sessions = set()  # (ptid, session_date) pairs with TR ≈ 3s
    for ptid_raw, sessions in data['subjects'].items():
        # Convert PTID format: "002S0295" -> "002_S_0295"
        if 'S' in ptid_raw:
            parts = ptid_raw.split('S')
            ptid = f"{parts[0]}_S_{parts[1]}"
        else:
            print(f"Skipping malformed PTID: {ptid_raw}")
            continue

        for session in sessions:
            session_id = session.get('session_id', '')
            if len(session_id) == 8:
                scan_records.append({
                    'PTID': ptid,
                    'ScanDate': pd.to_datetime(session_id, format='%Y%m%d'),
                    'TR_original': session.get('TR'),
                    'timepoints': session.get('timesteps')
                })
                tr = session.get('TR')
                if tr is not None and 2.9 <= tr <= 3.1:
                    session_date = f"{session_id[:4]}-{session_id[4:6]}-{session_id[6:8]}"
                    tr3_sessions.add((ptid, session_date))
            else:
                print(f"Skipping invalid session_id: {session_id}")
                continue

    print(f"Found {len(tr3_sessions)} sessions with TR ≈ 3s")

    df_scans = pd.DataFrame(scan_records)
    print(f"Loaded {len(df_scans)} valid scans from JSON.")

    # 2. Load Clinical Data (ADNIMERGE)
    adni_file = Path(os.environ.get("ADNI_MERGE_CSV", str(_META / "ADNIMERGE_14May2025.csv")))
    if not adni_file.exists():
        print(f"ERROR: '{adni_file}' not found.")
        return

    print("Loading ADNIMERGE...")
    cols = ['RID', 'PTID', 'VISCODE', 'EXAMDATE', 'DX', 'CDRSB', 'MMSE', 'AGE', 'PTGENDER', 'COLPROT']
    df_clinical = pd.read_csv(adni_file, usecols=cols, low_memory=False)
    df_clinical['EXAMDATE'] = pd.to_datetime(df_clinical['EXAMDATE'])
    df_clinical = df_clinical.dropna(subset=['EXAMDATE'])

    # 3. Fuzzy Match (90 Days Tolerance)
    print("Matching scans to clinical visits...")
    df_scans = df_scans.sort_values('ScanDate')
    df_clinical = df_clinical.sort_values('EXAMDATE')

    df_matched = pd.merge_asof(
        df_scans,
        df_clinical,
        left_on='ScanDate',
        right_on='EXAMDATE',
        by='PTID',
        direction='nearest',
        tolerance=pd.Timedelta('90 days')
    )

    df_matched = df_matched.dropna(subset=['RID'])
    print(f"Matched {len(df_matched)} scans to clinical data.")

    # 4. Calculate Prognosis FIRST (for all available MCI subjects)
    print("Calculating Prognosis for MCI subjects...")

    # Get all available subjects
    subj_ref = df_matched.sort_values('ScanDate').drop_duplicates(subset=['RID'], keep='first')
    subj_ref = subj_ref.dropna(subset=['DX'])

    # Calculate prognosis for MCI subjects
    full_history = df_clinical.sort_values('EXAMDATE')
    subject_prognosis = {}  # RID -> prognosis label

    for _, subj_row in subj_ref.iterrows():
        if subj_row['DX'] != 'MCI':
            continue

        rid = subj_row['RID']
        first_scan_date = subj_row['ScanDate']

        hist = full_history[full_history['RID'] == rid]
        future = hist[hist['EXAMDATE'] > first_scan_date]

        label = None
        conv = future[future['DX'] == 'Dementia']

        if not conv.empty:
            if (conv.iloc[0]['EXAMDATE'] - first_scan_date).days <= 1095:
                label = 'pMCI'

        if label is None and not future.empty:
            if (future.iloc[-1]['EXAMDATE'] - first_scan_date).days >= 1095:
                label = 'sMCI'

        if label:
            subject_prognosis[rid] = label

    # Add prognosis to subj_ref
    subj_ref['Prognosis'] = subj_ref['RID'].map(subject_prognosis)

    print(f"MCI subjects with prognosis: {subj_ref['Prognosis'].notna().sum()}")
    if subj_ref['Prognosis'].notna().any():
        print(subj_ref['Prognosis'].value_counts().to_dict())

    # 5. Create sample-level balanced dataset (1 sample per session for 500 timepoints)
    print("\nCreating balanced dataset for 500-timepoint samples...")
    print("Using 1 sample per session, prioritizing subject diversity...")

    SEED = 42

    # Get all sessions (not just subjects)
    sessions_pool = df_matched.copy()
    sessions_pool = sessions_pool.dropna(subset=['DX'])

    dx_counts = sessions_pool['DX'].value_counts()
    print(f"Available sessions by DX: {dx_counts.to_dict()}")

    # 1 session = 1 sample
    sessions_pool['samples_per_session'] = 1

    # Separate sessions into diagnostic groups (using .copy() to avoid warnings)
    cn_sessions = sessions_pool[sessions_pool['DX'] == 'CN'].copy()
    dem_sessions = sessions_pool[sessions_pool['DX'] == 'Dementia'].copy()
    mci_sessions = sessions_pool[sessions_pool['DX'] == 'MCI'].copy()

    print(f"\nSessions available:")
    print(f"CN: {len(cn_sessions)} sessions")
    print(f"Dementia: {len(dem_sessions)} sessions")
    print(f"MCI: {len(mci_sessions)} sessions")

    print(f"\nSessions per diagnostic group (1 session = 1 sample):")
    print(f"CN: {len(cn_sessions)} sessions")
    print(f"Dementia: {len(dem_sessions)} sessions")
    print(f"MCI: {len(mci_sessions)} sessions")

    # Find limiting factor for balanced samples (use session count since 1 sample per session)
    min_samples = min(len(cn_sessions), len(dem_sessions), len(mci_sessions))
    print(f"\nBalancing to {min_samples} sessions per diagnostic group")
    print(f"Total balanced sessions/samples: {min_samples * 3}")

    # Now select sessions to achieve balanced samples using greedy selection with metadata stratification
    def select_sessions_for_target_samples(sessions_df, target_samples, seed):
        """Select sessions to achieve target sample count, prioritizing subject diversity"""
        if len(sessions_df) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        np.random.seed(seed)

        # Prioritize unique subjects: select one session per subject first
        # Group by RID to identify subjects with multiple sessions
        sessions_by_subject = sessions_df.groupby('RID')

        # First pass: Select one session per subject (prioritize diversity)
        selected_sessions = []
        current_samples = 0
        used_subjects = set()

        # For each subject, pick their best session (most samples)
        for _, subject_sessions in sessions_by_subject:
            if current_samples >= target_samples:
                break

            # Pick the session with most samples per session for this subject
            best_session = subject_sessions.nlargest(1, 'samples_per_session').iloc[0]

            if current_samples + best_session['samples_per_session'] <= target_samples:
                selected_sessions.append(best_session)
                current_samples += best_session['samples_per_session']
                used_subjects.add(best_session['RID'])

        # Second pass: If we need more samples, add additional sessions from subjects already used
        if current_samples < target_samples:
            remaining_sessions = []
            for _, subject_sessions in sessions_by_subject:
                rid = subject_sessions.iloc[0]['RID']
                if rid in used_subjects:
                    # Get sessions we haven't selected yet
                    selected_indices = [s.name if hasattr(s, 'name') else None for s in selected_sessions]
                    additional = subject_sessions[~subject_sessions.index.isin(selected_indices)]
                    remaining_sessions.extend([row for _, row in additional.iterrows()])

            # Shuffle and add remaining sessions
            np.random.shuffle(remaining_sessions)
            for session in remaining_sessions:
                if current_samples + session['samples_per_session'] <= target_samples:
                    selected_sessions.append(session)
                    current_samples += session['samples_per_session']
                    if current_samples >= target_samples:
                        break

        selected_df = pd.DataFrame(selected_sessions)
        if len(selected_df) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        n_unique = len(selected_df['RID'].unique())
        print(f"  Selected {len(selected_df)} sessions from {n_unique} unique subjects ({current_samples} samples)")

        # Create stratification features for balanced splits
        selected_df_copy = selected_df.copy()
        selected_df_copy['age_bin'] = pd.cut(
            selected_df_copy['AGE'].fillna(70),
            bins=[0, 65, 75, 100],
            labels=['<65', '65-75', '>75']
        )
        selected_df_copy['gender_str'] = selected_df_copy['PTGENDER'].fillna('Unknown')

        strat_key = (
            selected_df_copy['age_bin'].astype(str) + '_' +
            selected_df_copy['gender_str'].astype(str)
        )

        strat_counts = strat_key.value_counts()
        min_strat_count = strat_counts.min()

        if min_strat_count >= 10:
            try:
                train_idx, test_val_idx = train_test_split(
                    selected_df.index,
                    test_size=0.2,
                    random_state=seed,
                    stratify=strat_key
                )

                test_val_df = selected_df.loc[test_val_idx]
                test_val_strat = strat_key.loc[test_val_idx]

                test_idx, val_idx = train_test_split(
                    test_val_df.index,
                    test_size=0.5,
                    random_state=seed,
                    stratify=test_val_strat
                )

                print("  Using stratified split for metadata balance")
                return (
                    selected_df.loc[train_idx],
                    selected_df.loc[test_idx],
                    selected_df.loc[val_idx]
                )
            except ValueError as e:
                print(f"  Warning: Stratification failed ({str(e)}), using simple split")
        else:
            print(f"  Warning: Insufficient samples per stratum (min={min_strat_count}), using simple split")

        n_total = len(selected_df)
        n_train = int(n_total * 0.80)
        n_test = int(n_total * 0.10)
        return (
            selected_df.iloc[:n_train],
            selected_df.iloc[n_train:n_train+n_test],
            selected_df.iloc[n_train+n_test:]
        )

    print(f"\nSelecting sessions to achieve {min_samples} sessions per group (prioritizing unique subjects)...")

    # Balance each diagnostic group
    cn_train, cn_test, cn_val = select_sessions_for_target_samples(cn_sessions, min_samples, SEED)
    dem_train, dem_test, dem_val = select_sessions_for_target_samples(dem_sessions, min_samples, SEED + 1)
    mci_train, mci_test, mci_val = select_sessions_for_target_samples(mci_sessions, min_samples, SEED + 2)

    # Combine all splits
    train_sessions = pd.concat([cn_train, dem_train, mci_train]) if len(cn_train) > 0 and len(dem_train) > 0 and len(mci_train) > 0 else pd.DataFrame()
    test_sessions = pd.concat([cn_test, dem_test, mci_test]) if len(cn_test) > 0 and len(dem_test) > 0 and len(mci_test) > 0 else pd.DataFrame()
    val_sessions = pd.concat([cn_val, dem_val, mci_val]) if len(cn_val) > 0 and len(dem_val) > 0 and len(mci_val) > 0 else pd.DataFrame()

    final_pool = pd.concat([train_sessions, test_sessions, val_sessions])

    # Add split labels
    if len(train_sessions) > 0:
        final_pool.loc[final_pool.index.isin(train_sessions.index), 'Split'] = 'Train'
    if len(test_sessions) > 0:
        final_pool.loc[final_pool.index.isin(test_sessions.index), 'Split'] = 'Test'
    if len(val_sessions) > 0:
        final_pool.loc[final_pool.index.isin(val_sessions.index), 'Split'] = 'Val'

    print(f"\n=== Final Selection Summary ===")
    print(f"Total sessions selected: {len(final_pool)}")

    if len(train_sessions) > 0:
        print(f"  Train: {len(train_sessions)} sessions from {train_sessions['RID'].nunique()} subjects")
    if len(test_sessions) > 0:
        print(f"  Test: {len(test_sessions)} sessions from {test_sessions['RID'].nunique()} subjects")
    if len(val_sessions) > 0:
        print(f"  Val: {len(val_sessions)} sessions from {val_sessions['RID'].nunique()} subjects")

    target_rids = final_pool['RID'].unique().tolist() if len(final_pool) > 0 else []
    print(f"  Total unique subjects: {len(target_rids)}")

    # 6. Build prognosis map for all scans
    print("Building prognosis map for all scans...")
    prognosis_map = {}

    for _, scan_row in final_pool.iterrows():
        if scan_row['DX'] != 'MCI': continue

        rid = scan_row['RID']
        scan_date = scan_row['ScanDate']

        hist = full_history[full_history['RID'] == rid]
        future = hist[hist['EXAMDATE'] > scan_date]

        label = None
        conv = future[future['DX'] == 'Dementia']

        if not conv.empty:
            if (conv.iloc[0]['EXAMDATE'] - scan_date).days <= 1095:
                label = 'pMCI'

        if label is None and not future.empty:
            if (future.iloc[-1]['EXAMDATE'] - scan_date).days >= 1095:
                label = 'sMCI'

        if label:
            prognosis_map[(rid, scan_date)] = label

    # 7. Build Output
    output_data = []
    for _, row in final_pool.iterrows():
        split = row.get('Split', 'Unknown')  # Use the split that was already assigned

        prog = prognosis_map.get((row['RID'], row['ScanDate']), np.nan)

        # Diagnosis Label (kept separate from Current_DX for clarity)
        label_diag = np.nan
        if row['DX'] == 'CN': label_diag = 'CN'
        elif row['DX'] == 'Dementia': label_diag = 'Dementia'

        output_data.append({
            'RID': row['RID'],
            'PTID': row['PTID'],
            'VISCODE_Clinical': row['VISCODE'],
            'EXAMDATE_Clinical': row['EXAMDATE'].strftime('%Y-%m-%d'),
            'SCANDATE_Imaging': row['ScanDate'].strftime('%Y-%m-%d'),
            'Days_Diff': (row['ScanDate'] - row['EXAMDATE']).days,
            'Split': split,
            'Current_DX': row['DX'],
            'Label_Diag_AD_vs_CN': label_diag,
            'Label_Prog_pMCI_vs_sMCI': prog,
            'Score_CDRSB': row['CDRSB'],
            'Score_MMSE': row['MMSE'],
            'AGE': row['AGE'],
            'PTGENDER': row['PTGENDER']
        })

    final_df = pd.DataFrame(output_data)

    # Clean empty rows - require Current_DX to be present
    final_df_clean = final_df.dropna(subset=['Current_DX'])

    # Additionally drop rows where ALL other targets are missing
    targets = ['Label_Prog_pMCI_vs_sMCI', 'Score_CDRSB', 'Score_MMSE']
    final_df_clean = final_df_clean.dropna(subset=targets, how='all')

    # Filter to TR ≈ 3s sessions only
    n_before = len(final_df_clean)
    final_df_clean = final_df_clean[
        final_df_clean.apply(
            lambda r: (r['PTID'], r['SCANDATE_Imaging']) in tr3_sessions, axis=1
        )
    ]
    print(f"TR filter: {n_before} → {len(final_df_clean)} rows (kept TR ≈ 3s sessions)")

    outfile = _META / "adni_fmri_benchmark_split.csv"
    final_df_clean.to_csv(outfile, index=False)

    print(f"Done. Saved {len(final_df_clean)} rows to {outfile}")


if __name__ == "__main__":
    generate_benchmark_from_json()
