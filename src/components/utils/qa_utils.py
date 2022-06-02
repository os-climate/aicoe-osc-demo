import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_excels(annotation_folder, columns_to_read):
    excels = [f for f in os.listdir(annotation_folder) if f.endswith('.xlsx')]

    dfs = []

    for f in excels:
        fname = os.path.join(annotation_folder, f)
        df = pd.read_excel(fname, header=0)
        cols = df.columns
        assert(all([e in cols for e in columns_to_read])), \
                "{} doesn't have certain columns {}".format(f, columns_to_read)


        if 'Sector' in cols:
            df.rename({'Sector':'sector'}, axis=1, inplace=True)
            columns_to_read = columns_to_read + ['sector']
        elif 'sector' in cols:
            columns_to_read = columns_to_read + ['sector']
        else:
            logger.info("{} has no column Sector/sector".format(f))

        if 'annotator' in cols:
            columns_to_read += ['annotator']
        else:
            df['annotator'] = f

        df = df[columns_to_read]
        dfs.append(df)

    df = pd.concat(dfs)

    return df


def clean_annotation(df, save_path, kpi_category, exclude=['CEZ']):
    """ Returns a clean dataframe and save it after
        1. dropping all NaN rows
        2. dropping rows which has NaN values in some of the columns
        (refer below)
        3. remove template company
        4. cleaning source_file column
        5. clean data_type column
        6. clean source_page column
        7. remove examples with wrong (kpi, data_type)
    Args:
        df (A dataframe)
        save_dir (A path)
        exclude (A list of str): Companies to exclude
    """
    # dropping all nan rows
    df = df.dropna(axis=0, how='all').reset_index(drop=True)

    # Drop rows with NaN for any of these columns except answer, relevant_paragraphs
    df = df.dropna(
        axis=0,
        how='any',
        subset=['company', 'source_file', 'source_page', 'kpi_id', 'year']
    ).reset_index(drop=True)

    # Remove template company
    if exclude != []:
        df = df[~df.company.isin(exclude)]

    # Get pdf filename right (don't need to make it a class method)
    def get_pdf_name_right(f):
        if not f.endswith('.pdf'):
            if f.endswith(',pdf'):
                filename = f.split(',pdf')[0].strip() + '.pdf'
            else:
                filename = f.strip() + '.pdf'
        else:
            filename = f.split('.pdf')[0].strip() + '.pdf'

        return filename

    df['source_file'] = df['source_file'].apply(get_pdf_name_right)

    # clean data type
    df['data_type'] = df['data_type'].apply(str.strip)

    # Remove examples where source_page can't be parsed
    def clean_page(sp):
        if sp[0] != "[" or sp[-1] != "]":
            return None
        else:
            # Covers multi pages and fix cases like '02'
            return [str(int(i)) for i in sp[1:-1].split(",")]

    temp = df['source_page'].apply(clean_page)
    invalid_source_page = df['source_page'][temp.isna()].unique().tolist()
    if len(invalid_source_page) != 0:
        logger.warning(
            "Has invalid source_page format: {} and {} such examples".format(
                df['source_page'][temp.isna()].unique(),
                df['source_page'][temp.isna()].shape[0]
            )
        )

    df['source_page'] = temp
    df = df.dropna(axis=0, subset=['source_page']).reset_index(drop=True)

    # Remove examples with incorrect kpi-data_type pair
    def clean_id(r):
        try:
            kpi_id = float(r['kpi_id'])
        except ValueError:
            kpi_id = r['kpi_id']

        try:
            if r['data_type'] in kpi_category[kpi_id]:
                cat = True
            else:
                cat = False
        except KeyError:
            cat = True

        return cat

    correct_id_bool = df[['kpi_id', 'data_type']].apply(clean_id, axis=1)
    df = df[correct_id_bool].reset_index(drop=True)
    diff = correct_id_bool.shape[0] - df.shape[0]
    if diff > 0:
        logger.info(
            "Drop {} examples due to incorrect kpi-data_type pair"\
            .format(diff)
        )

    df.to_csv(save_path)
    logger.info("{} is created.".format(save_path))

    return df