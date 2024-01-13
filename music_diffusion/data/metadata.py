# -*- coding: utf-8 -*-
import re
from glob import glob
from os.path import basename, join
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def __read_scoring_legend(scoring_df: pd.DataFrame) -> Dict[str, str]:
    voices_df = (
        scoring_df.loc[:1]
        .transpose()
        .reset_index()[["level_1", 0]]
        .rename({"level_1": "code", 0: "scoring"}, axis=1)
    )

    winds_battery_df = scoring_df.loc[2:3]
    winds_battery_df.columns = winds_battery_df.iloc[0]
    winds_battery_df = (
        winds_battery_df.reset_index()
        .drop("index", axis=1)
        .drop(index=0)
        .transpose()
        .reset_index()
        .rename({2: "code", 1: "scoring"}, axis=1)
    )

    strings_keyboard_df = scoring_df.loc[5:6]
    strings_keyboard_df.columns = strings_keyboard_df.iloc[0]
    strings_keyboard_df = (
        strings_keyboard_df.reset_index()
        .drop("index", axis=1)
        .drop(index=0)
        .transpose()
        .reset_index()
        .drop(index=0)
        .rename({5: "code", 1: "scoring"}, axis=1)
    )

    scoring_legend_df = (
        pd.concat([voices_df, winds_battery_df, strings_keyboard_df], axis=0)
        .reset_index()
        .drop("index", axis=1)
    )

    scoring_legend_dict = {
        row["code"]: row["scoring"] for _, row in scoring_legend_df.iterrows()
    }

    return scoring_legend_dict


def __extract_flac_bvw(
    bach_complete_works_path: str, bach_works_df: pd.DataFrame
) -> pd.DataFrame:
    musics = glob(
        join(bach_complete_works_path, "**", "*.flac"), recursive=True
    )

    regex_music = re.compile(r"^.* BWV (\d+[a-z]?).*\.flac$")

    music_bwv = {}

    for music in musics:
        filename = basename(music)
        if regex_music.match(filename):
            matched_bwv = regex_music.match(filename)
            if matched_bwv:
                music_bwv[music] = matched_bwv.group(1)

    bwv_set = set(bach_works_df["BWV_without_version"])

    music_path_df = pd.DataFrame(
        [
            [wav_path, bwv]
            for wav_path, bwv in music_bwv.items()
            if bwv in bwv_set
        ],
        columns=["wav_path", "bwv"],
    )

    return music_path_df


def __parse_scoring(bach_works_df: pd.DataFrame) -> pd.Series:
    regex_voices = re.compile(
        r"^((?:[satbSATBvV\d?()]|(?:[vV]\.[12]))+)(?: .+)?$"
    )
    regex_voices_2 = re.compile(r"[satbSATBvV]|(?:[vV]\.[12])")

    regex_scoring = re.compile(
        r"^(?:(?:[satbSATBvV?\d()]|[vV]\.[12])+ )?(.+)$"
    )
    # regex_scoring_number = re.compile(r"(\d+)")
    regex_scoring_3 = re.compile(r"([A-z]+)")
    regex_scoring_4 = re.compile(r"([1-9]*[A-z]+)")

    remove_scoring = {"SBBB", "Nho", "colla", "parte", "instr", "or", "mezzo"}
    rename_scoring_dict = {
        "Harpsichord": "Hc",
        "Vla": "Va",
        "Vlp": "Vl",
        "Tne": "Tbn",
        "Keyboard": "Kb",
        "Fag": "Bas",
        "Cdc": "Hn",
        "Organ": "Org",
        "Gam": "Vdg",
    }

    def __parse_one_scoring(s: str) -> List[str]:

        scoring = []

        matched_voices = regex_voices.match(s)

        has_matched_voices = False
        if matched_voices:
            for grp in regex_voices_2.findall(matched_voices.group(1)):
                scoring.append(grp)
                has_matched_voices = True

        matched_scoring = regex_scoring.match(s)

        if matched_scoring and not (
            len(regex_scoring_4.findall(s)) == 1 and has_matched_voices
        ):
            for grp in matched_scoring.group(1).split(" "):
                # number = regex_scoring_2.search(grp)
                # number = 1 if not number else int(number.group(1))

                found_sco = regex_scoring_3.search(grp)
                if found_sco and found_sco.group(1) not in remove_scoring:
                    scoring.append(
                        rename_scoring_dict[found_sco.group(1)]
                        if found_sco.group(1) in rename_scoring_dict
                        else found_sco.group(1)
                    )

        return scoring

    return bach_works_df["Scoring"].fillna("").apply(__parse_one_scoring)


def __get_genre(bach_works_df: pd.DataFrame) -> pd.Series:
    metadata_url = "https://www.bachdigital.de/receive/BachDigitalWork_work_"

    def __get_one_genre(url: str) -> Optional[str]:
        page = BeautifulSoup(
            requests.get(url, timeout=5).content, features="lxml"
        )
        dl = page.find("dl", {"id": "generalData"})
        if dl:
            dts = dl.find_all("dt")
            for dt in dts:
                if dt.get_text().replace("\n", "") == "Genre":
                    genre: str = dt.find_next("dd").get_text()
                    return genre
        return None

    tqdm.pandas()

    return (
        metadata_url
        + bach_works_df["BD"].str.pad(9, fillchar="0", side="left")
        + "?lang=en"
    ).progress_apply(__get_one_genre)


def create_metadata_csv(
    bach_complete_works_path: str,
    output_metadata_csv_path: str,
    # scoring_legend_csv_path: str
) -> None:
    html_dfs = pd.read_html(
        "https://en.wikipedia.org/wiki/"
        "List_of_compositions_by_Johann_Sebastian_Bach",
        flavor="bs4",
    )

    # Read Bach works

    bach_works_df = html_dfs[5].copy()

    # remove unreferenced
    bach_works_df = bach_works_df[~bach_works_df["BWV"].isna()]
    # remove variants
    bach_works_df = bach_works_df[
        ~bach_works_df["BWV"].str.contains("/", regex=False)
    ]
    # remove html table separations
    bach_works_df = bach_works_df[
        bach_works_df["BD"].fillna("").str.match(r"^\d+$")
    ]

    bach_works_df["BWV_without_version"] = bach_works_df["BWV"].str.extract(
        r"^(\d+)"
    )
    bach_works_df["version"] = (
        bach_works_df["BWV"].str.extract(r"^\d+\.(\d+)").fillna(1).astype(int)
    )

    bach_works_df = bach_works_df.loc[
        bach_works_df.groupby("BWV_without_version")["version"].idxmax()
    ]

    # Read all flac
    music_path_df = __extract_flac_bvw(bach_complete_works_path, bach_works_df)

    # read scoring legend
    # scoring_legend_dict = __read_scoring_legend(html_dfs[6].copy())

    # create scoring column
    bach_works_df["scoring"] = __parse_scoring(bach_works_df)

    # create genre column
    bach_works_df["genre"] = __get_genre(bach_works_df)
    bach_works_df = bach_works_df[~bach_works_df["genre"].isna()]

    # rename columns
    bach_works_df = bach_works_df[
        ["Name", "Key", "BWV_without_version", "scoring", "genre"]
    ].rename(
        columns={
            "Name": "name",
            "Key": "key",
            "BWV_without_version": "bwv",
        }
    )

    # fix space character
    bach_works_df["key"] = bach_works_df["key"].str.replace("\xa0", " ")
    bach_works_df["name"] = bach_works_df["name"].str.replace("\xa0", " ")

    # join with music flac paths
    final_df = music_path_df.merge(bach_works_df, on=["bwv"], how="inner")

    final_df.to_csv(output_metadata_csv_path, sep=";", index=False)
