"""
Command Line Interface
"""

import os

import pandas as pd

from pandas.io.stata import StataReader, StataWriter117
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from Helpers import arg_parser

def execute_main()->None:

    args = arg_parser()
    args_dict = vars(args)

    print(args_dict.keys())

    input_file   = args_dict['input_file']
    output_folder= args_dict['output_folder']
    ids_2_select = args_dict['ids_to_select']

    # check paths
    if not os.path.exists(input_file):
        raise FileNotFoundError("Input file cannot be found.")
    
    if not os.path.exists(output_folder):
        raise FileNotFoundError("Output folder cannot be found.")
    
    # imports
    from SampleId import idKeepAlphanumeric, idFormatMarker, idAddZero, idMiddleNull, idZeroSubstitute
    from InfoId import centreExtractor, ControlStatus
    from Age import AgeCorrector, BasicAgeImputer
    from DiseaseDur import DiseaseDuration, AgeOnset
    from SympDur import SymptomDuration, SymptomDurationFixer, SymptomOnsetFixer
    from Education import EducationStandandizer, ExtractEducation, EducationSubstition, EducationMissing, EducationExtreme
    from Other import AssessmentDate, HeightWeight, BMICalculator, RecodeCenters
    from MOCA import CleanerMOCA, FormatMOCA, ExtremesMOCA
    from Extremes import ExtremesYears, ExtremeValues, DateFixer, FormatBDI, FormatString2Int
    from Other import Move2Other, ClassifyOnset, ClassifyEducation, RecodeGeography, FromUPDRStoMDS
    
    from Helpers import recover_columns_names, detect_datetime_cols, select_value_labels, rearrange_columns

    date_cols = [
    'date_of_assessment', 'date', 'retirement_year', 'start_date', 'start_date_2', 'start_date_3',
    'start_date_4', 'start_date_6', 'start_date_7', 'start_date_9', 'start_date_10', 'start_date_11',
    'start_date_12', 'start_date_13', 'date_stored'
    ]

    stata = StataReader(input_file)

    df_input = stata.read(
        preserve_dtypes     =False, 
        convert_categoricals=False, 
        convert_dates       =True
    )

    cols_to_report = ['participant_id', 'record_id', 'redcap_data_access_group']

    idClean_pipe = Pipeline([
    ('alphaNum', idKeepAlphanumeric().set_output(transform='pandas')),
    ('zerosSubs', idZeroSubstitute().set_output(transform='pandas')),
    ('middleZero', idMiddleNull().set_output(transform='pandas')),
    ('addZero', idAddZero().set_output(transform='pandas')),
    ('centreExtract', centreExtractor().set_output(transform='pandas')),
    ('controlStatus', ControlStatus(outputCol='Status').set_output(transform='pandas')),
    ('formatCheck', idFormatMarker(outputCol='no_ID_prob').set_output(transform='pandas'))
    ])
    age_pipe = Pipeline([
        ('ageCorrector', AgeCorrector().set_output(transform='pandas'))
    ])
    sympton_pipe = Pipeline([
        ('symptomTime', SymptomDuration().set_output(transform='pandas'))
    ])
    pdDuration_pipe = Pipeline([
        ('pdDuration', SymptomDuration().set_output(transform='pandas'))
    ])
    moca_pipe = Pipeline([
        ('cleanmoca', CleanerMOCA().set_output(transform='pandas')),
        ('formatmoca', FormatMOCA().set_output(transform='pandas')),
        ('extrememoca', ExtremesMOCA().set_output(transform='pandas'))
    ])
    years_pipe = Pipeline([
        ('extremYears', ExtremesYears(earlier_year=1990, later_year=2024).set_output(transform='pandas'))
    ])
    cognImpDur_pipe = Pipeline([
        ('cognImpDur', SymptomDuration().set_output(transform='pandas')),
        ('extremValCogn', ExtremeValues().set_output(transform='pandas'))
    ])
    dates_pipe = Pipeline([
        ('dates', DateFixer().set_output(transform='pandas'))
    ])
    bdi_pipe = Pipeline([
        ('bdiStages', FormatBDI().set_output(transform='pandas'))
    ])
    updrsI_pipe = Pipeline([
        ('updrsIpipe', FormatString2Int(splitter='/', lower_bound=0, upper_bound=16).set_output(transform='pandas'))
    ])
    updrsII_pipe = Pipeline([
        ('updrsIIpipe', FormatString2Int(splitter='/', lower_bound=0, upper_bound=52).set_output(transform='pandas'))
    ])
    updrsIII_pipe = Pipeline([
        ('updrsIIIpipe', FormatString2Int(splitter='/', lower_bound=0, upper_bound=56).set_output(transform='pandas'))
    ])
    updrsIVa_pipe = Pipeline([
        ('updrsIVapipe', FormatString2Int(splitter='/', lower_bound=0, upper_bound=13).set_output(transform='pandas'))
    ])
    updrsIVb_pipe = Pipeline([
        ('updrsIVbpipe', FormatString2Int(splitter='/', lower_bound=0, upper_bound=7).set_output(transform='pandas'))
    ])
    updrsIVc_pipe = Pipeline([
        ('updrsIVcpipe', FormatString2Int(splitter='/', lower_bound=0, upper_bound=3).set_output(transform='pandas'))
    ])
    mdsI_pipe = Pipeline([
        ('mdsIpipe', FormatString2Int(splitter=' ', lower_bound=0, upper_bound=52).set_output(transform='pandas'))
    ])
    mdsII_pipe = Pipeline([
        ('mdsIIpipe', FormatString2Int(splitter=' ', lower_bound=0, upper_bound=52).set_output(transform='pandas'))
    ])
    mdsIII_pipe = Pipeline([
        ('mdsIIIpipe', FormatString2Int(splitter=' ', lower_bound=0, upper_bound=132).set_output(transform='pandas'))
    ])
    mdsIV_pipe = Pipeline([
        ('mdsIVpipe', FormatString2Int(splitter=' ', lower_bound=0, upper_bound=24).set_output(transform='pandas'))
    ])
    natWork_pipe = Pipeline([
        ('nature', Move2Other(feature_name='nature_of_work').set_output(transform='pandas'))
    ])
    centers_pipe = Pipeline([
        ('center', RecodeCenters().set_output(transform='pandas'))
    ])

    years_onset_columns = [
    'falls_year_of_onset', 'falls_year_of_onse_2', 'falls_year_of_onse_3', 'falls_year_of_onse_4',
    'cognitive_impairment_yoa', 'falls_year_of_onse_6', 'falls_year_of_onse_7', 'falls_year_of_onse_8',
    'falls_year_of_onse_9', 'pd_or_parkinsonism_year_of'
    ]

    cleaning_trans = ColumnTransformer([
    ('idChecker', idClean_pipe, ['participant_id']),
    ('Age', age_pipe, ['age']),
    ('SymptomDur', sympton_pipe, ['initial_symptom_duration_u']),
    ('ParkinsonDur', pdDuration_pipe, ['pd_or_parkinsonism_duratio']),
    ('mocaClean', moca_pipe, ['total_score_for_moca']),
    ('yearOnset', years_pipe, years_onset_columns),
    ('cgnYear', years_pipe, ['cognitive_impairment_year']),
    ('cogImpDur', cognImpDur_pipe, ['cognative_impairment_durat']),
    ('dateFixer', dates_pipe, date_cols),
    ('bdi', bdi_pipe, ['total_score_for_bdi']),
    ('updrsI', updrsI_pipe, ['updrs_1_total_score']),
    ('updrsII', updrsII_pipe, ['updrs_part_ii_total_score']),
    ('updrsIII', updrsIII_pipe, ['updrs_part_iii_total_score']),
    ('updrsIVa', updrsIVa_pipe, ['updrs_part_iv_a_total_scor']),
    ('updrsIVb', updrsIVb_pipe, ['updrs_part_iv_b_total_scor']),
    ('updrsIVc', updrsIVc_pipe, ['updrs_part_iv_c_total_scor']),
    ('mdsI', mdsI_pipe, ['total_score_part_i']),
    ('mdsII', mdsII_pipe, ['total_score_part_ii']),
    ('mdsIII', mdsIII_pipe, ['total_score_part_iii']),
    ('mdsIV', mdsIV_pipe, ['total_score_part_iv']),
    ('center_rec', centers_pipe, ['redcap_data_access_group']),
    ('natWork', natWork_pipe, ['nature_of_work___1', 'nature_of_work___2', 'nature_of_work___3', 'nature_of_work___4'])
    ],
    remainder='passthrough').set_output(transform='pandas')

    df_1 = cleaning_trans.fit_transform(df_input)
    df_1.columns = recover_columns_names(df_1.columns)
    df_1

    # =======================================================================
    #               FILTER
    # =======================================================================

    df_IDs = df_1[~df_1['no_ID_prob']][cols_to_report].reset_index(drop=True)
    df_IDs.to_csv(os.path.join(output_folder, 'problematic_id.csv'))

    df_ID_dup = df_1[df_1['participant_id']\
                    .duplicated(keep=False)][cols_to_report]\
                    .copy().sort_values('participant_id')\
                    .reset_index(drop=True)
    df_ID_dup.to_csv(os.path.join(output_folder, 'duplicated_id.csv'))

    # to stay, filtering process
    df_1 = df_1[df_1['no_ID_prob']].reset_index(drop=True).drop(columns=['no_ID_prob', 'control'])
    df_1 = df_1[~df_1['participant_id'].duplicated(keep=False)].reset_index(drop=True)

    to_estimate_mds = [
    'hoehn_and_yahr_staging', 'updrs_1_total_score', 'updrs_part_ii_total_score',
    'updrs_part_iii_total_score', 'updrs_part_iv_a_total_scor', 'updrs_part_iv_b_total_scor',
    'updrs_part_iv_c_total_scor'
    ]

    assess_pipe = Pipeline([
        ('assessDate', AssessmentDate().set_output(transform='pandas'))
    ])
    estimate_mds = Pipeline([
        ('estim_mds', FromUPDRStoMDS(outputStr='estim_MDS').set_output(transform='pandas'))
    ])
    assess_trans = ColumnTransformer([
        ('assessm', assess_pipe, ['participant_id', 'date_of_assessment']),
        ('estimate_MDS', estimate_mds, to_estimate_mds)
    ],
    remainder='passthrough').set_output(transform='pandas')

    df_2 = assess_trans.fit_transform(df_1)
    df_2.columns = recover_columns_names(df_2)
    df_2

    ageImp_pipe = Pipeline([
    ('ageImputer', BasicAgeImputer().set_output(transform='pandas'))
    ])
    hw_pipe = Pipeline([
        ('hwFixer', HeightWeight().set_output(transform='pandas'))
    ])
    ageHW_trans = ColumnTransformer([
        ('ageImp', ageImp_pipe, ['age', 'date', 'date_of_assessment']),
        ('hwTrans', hw_pipe, ['height', 'weight'])
    ],
    remainder='passthrough').set_output(transform='pandas')

    df_3 = ageHW_trans.fit_transform(df_2)
    df_3.columns = recover_columns_names(df_3)
    df_3

    bmi_pipe = Pipeline([
    ('bmiComp', BMICalculator(outputCol='bmi_comp').set_output(transform='pandas'))
    ])
    ageExt_pipe = Pipeline([
        ('ageExt', ExtremeValues(lower_bound=17, upper_bound=97).set_output(transform='pandas'))
    ])
    symFix_pipe = Pipeline([
        ('symFixer', SymptomDurationFixer(upper_bound=30, keep_assess_date=True).set_output(transform='pandas'))
    ])
    PDdurFix_pipe = Pipeline([
        ('pdFixer', SymptomDurationFixer(upper_bound=30, keep_assess_date=False).set_output(transform='pandas'))
    ])
    yearSym_pipe = Pipeline([
        ('yearFirst', SymptomOnsetFixer(lower_bound=1990).set_output(transform='pandas'))
    ])
    bmi_trans = ColumnTransformer([
        ('bmi', bmi_pipe, ['height', 'weight']),
        ('symptom', symFix_pipe, ['initial_symptom_duration_u', 'date_of_assessment']),
        ('PDfixer', PDdurFix_pipe, ['pd_or_parkinsonism_duratio', 'date_of_assessment']),
        ('firstFix', yearSym_pipe, ['initial_symptom_year_of_on']),
        ('ageExtr', ageExt_pipe, ['age'])
    ],
    remainder='passthrough').set_output(transform='pandas')

    df_4 = bmi_trans.fit_transform(df_3)
    df_4.columns = recover_columns_names(df_4)
    df_4

    # ==========================================================================

    diseaseDur_pipe = Pipeline([
        ('disease', DiseaseDuration(outputCol='PD_duration').set_output(transform='pandas'))
    ])
    diseaseDur_columns = ['date_of_assessment', 'initial_symptom_year_of_on', 'initial_symptom_duration_u', 
                          'pd_or_parkinsonism_duratio', 'pd_or_parkinsonism_year_of']
    disease_trans = ColumnTransformer([
        ('diseaseDur', diseaseDur_pipe, diseaseDur_columns),
    ],
    remainder='passthrough').set_output(transform='pandas')

    df_5 = disease_trans.fit_transform(df_4)
    df_5.columns = recover_columns_names(df_5)
    df_5

    edu_0_years = ['uneducated', 'illiterate', 'none', 'noteducated', 'no', 'illititerate', 
                   'illilerate', 'no(uneducated)', 'uneducate', 'na(uneducated)']

    edu_pipeline = Pipeline([
        ('eduStd', EducationStandandizer().set_output(transform='pandas')),
        ('extracter', ExtractEducation().set_output(transform='pandas')),
        ('subst_0', EducationSubstition(cats_list=edu_0_years, num_years=0).set_output(transform='pandas')),
        ('nulls_sub', EducationMissing().set_output(transform='pandas')),
        ('extremes', EducationExtreme(max_allowed_val=50).set_output(transform='pandas'))
    ])

    ageOnset_pipe = Pipeline([
        ('ageOnset', AgeOnset(outputCol='age_at_onset').set_output(transform='pandas'))
    ])
    onset_trans = ColumnTransformer([
        ('education', edu_pipeline, ['years_of_education']),
        ('ageonset', ageOnset_pipe, ['age', 'PD_duration'])
    ],
    remainder='passthrough').set_output(transform='pandas')

    df_6 = onset_trans.fit_transform(df_5)
    df_6.columns = recover_columns_names(df_6)
    df_6

    # ===========================================================
    #          TESTING



    onset_type = Pipeline([
        ('typeOnset', ClassifyOnset(outputCol='onset_type').set_output(transform='pandas'))
    ])
    edu_level = Pipeline([
        ('eduLevel', ClassifyEducation(outputCol='education_level').set_output(transform='pandas'))
    ])
    geo_trans = Pipeline([
        ('geo_reco', RecodeGeography(outputCol='zone_of_origin').set_output(transform='pandas'))
    ])

    class_trans = ColumnTransformer([
        ('edu_level', edu_level, ['years_of_education']),
        ('onset_type', onset_type, ['age_at_onset']),
        ('geoReco', geo_trans, ['state_of_origin'])
    ],
    remainder='passthrough').set_output(transform='pandas')

    df_7 = class_trans.fit_transform(df_6)
    df_7.columns = recover_columns_names(df_7)
    df_7

    date_dict = detect_datetime_cols(df_7)

    labels = select_value_labels(df_7, stata.value_labels())

    labels['Status'] = {1: 'Patient', 0:'Control'}
    labels['onset_type'] = {0: 'Juvenile Onset', 1:'Young Onset', 2:'Late Onset'}
    labels['education_level'] = {0: 'Illiterate', 1:'1 to 7', 2:'8 to 12', 3:'Above 12'}
    labels['zone_of_origin'] = {1:'Northern Zone', 2: 'Central Zone', 3: 'Eastern Zone', 4: 'Western Zone', 5: 'Southern Zone', 6: 'North Eastern Zone'}

    df_7 = df_7[rearrange_columns(df_7, df_input.columns.to_list())].copy()

    writer = pd.io.stata.StataWriter117(os.path.join(output_folder, 'cleaned_data.dat'), df_7, 
                                        value_labels  =labels, 
                                        convert_dates =date_dict,
                                        variable_labels=stata.variable_labels())
    writer.write_file()

    if ids_2_select is not None:

        df_IDS = pd.read_csv(ids_2_select, index_col=False)
        df= df_7.merge(df_IDS, on=df_IDS.columns[0])
        writer_sel = pd.io.stata.StataWriter117(
            os.path.join(output_folder, 'selected_data.dat'), df,
            value_labels  =labels, 
            convert_dates =date_dict,
            variable_labels=stata.variable_labels()
        )
        writer_sel.write_file()

    return None

if __name__ == "__main__":
    execute_main()
