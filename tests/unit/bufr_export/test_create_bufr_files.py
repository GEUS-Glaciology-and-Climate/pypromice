from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest import TestCase

import toml
from pypromice.station_configuration import write_station_configuration_mapping

from pypromice.postprocess.create_bufr_files import create_bufr_files
from tests.utilities import get_station_configuration

DATA_DIR = Path(__file__).parent.absolute()


def create_data_file(path: Path, src_path: Optional[Path] = None):
    if src_path is None:
        src_path = Path("/dev/null")

    path.parent.mkdir(exist_ok=True, parents=True)
    with src_path.open() as fp_src:
        with path.open("w") as fp_out:
            fp_out.write(fp_src.read())


class TestCreateBufrFiles(TestCase):
    def setUp(self):
        self._temp_dir = TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_create_bufr_files(self):
        """
        Test the creation of bufr files and their output folder structure.
        It does not test the content of the bufr files.
        """
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"
        input_files = [
            input_dir / "THU_L2_hourly.csv",
            input_dir / "KAN_Lv3_hourly.csv",
        ]
        # Use the same data for all input files
        for input_file in input_files:
            create_data_file(
                input_file,
                src_path=DATA_DIR.joinpath("tx_l3_test1.csv"),
            )

        station_configuration_root = self.temp_dir / "station_configuration"
        station_configuration_root.mkdir(parents=True, exist_ok=True)
        station_configuration_mapping = {
            "THU_L2": get_station_configuration(stid="THU_L2", export_bufr=True),
            "KAN_Lv3": get_station_configuration(stid="KAN_Lv3", export_bufr=True),
        }
        write_station_configuration_mapping(
            station_configurations=station_configuration_mapping,
            configuration_root_dir=station_configuration_root,
        )

        create_bufr_files(
            input_files=input_files,
            period_start="2023-12-06T00:00",
            period_end="2023-12-06T04:00",
            output_root=output_dir,
            override=True,
            break_on_error=True,
            station_configuration_root=station_configuration_root,
        )

        compiled_output_dir = output_dir / "compiled"
        individual_output_root = output_dir / "individual"
        self.assertTrue(compiled_output_dir.exists())
        self.assertTrue(individual_output_root.exists())
        expected_output_timestamps = [
            "20231206T0000",
            "20231206T0100",
            "20231206T0200",
            "20231206T0300",
            "20231206T0400",
        ]
        compiled_output_files = sorted(compiled_output_dir.glob("*.bufr"))
        expected_output_file_names = sorted(
            [
                f"geus_{timestamp_str}.bufr"
                for timestamp_str in expected_output_timestamps
            ]
        )
        self.assertListEqual(
            expected_output_file_names, [p.name for p in compiled_output_files]
        )
        individual_output_dirs = sorted(individual_output_root.glob("*"))
        self.assertListEqual(
            expected_output_timestamps, [p.stem for p in individual_output_dirs]
        )
        for dir in individual_output_dirs:
            # There should be a bufr file for each station
            self.assertTrue((dir / "THU_L2.bufr").exists())
            self.assertTrue((dir / "KAN_Lv3.bufr").exists())

    def test_get_bufr_from_empty_data_file_raises_error(self):
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"
        input_file = input_dir / "THU_L2_hourly.csv"
        create_data_file(input_file, src_path=None)
        station_configuration_root = self.temp_dir / "station_configuration"
        station_configuration = get_station_configuration(
            stid="KAN_Lv3", export_bufr=True
        )
        write_station_configuration_mapping(
            station_configurations={station_configuration.stid: station_configuration},
            configuration_root_dir=station_configuration_root,
        )

        with self.assertRaises(ValueError):
            create_bufr_files(
                input_files=[input_file],
                period_start="2023-12-06T00:00",
                period_end="2023-12-06T04:00",
                output_root=output_dir,
                override=True,
                break_on_error=True,
                station_configuration_root=station_configuration_root,
            )

    def test_get_bufr_continues_when_break_on_error_is_false(self):
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"
        input_file_without_data = input_dir / "THU_L2_hourly.csv"
        create_data_file(input_file_without_data, src_path=None)
        input_file_with_data = input_dir / "KAN_Lv3_hourly.csv"
        create_data_file(
            input_file_with_data, src_path=DATA_DIR.joinpath("tx_l3_test1.csv")
        )
        compiled_output_dir = output_dir / "compiled"
        individual_output_root = output_dir / "individual"
        station_configuration_root = self.temp_dir / "station_configuration"
        write_station_configuration_mapping(
            station_configurations={
                "THU_L2": get_station_configuration(stid="THU_L2", export_bufr=True),
                "KAN_Lv3": get_station_configuration(stid="KAN_Lv3", export_bufr=True),
            },
            configuration_root_dir=station_configuration_root,
        )
        expected_compiled_output_file = compiled_output_dir / "geus_20231206T0000.bufr"
        expected_individual_output_dir = individual_output_root / "20231206T0000"
        expected_individual_output_file = (
            expected_individual_output_dir / "KAN_Lv3.bufr"
        )

        create_bufr_files(
            input_files=[
                input_file_without_data,
                input_file_with_data,
            ],
            period_start="2023-12-06T00:00",
            period_end="2023-12-06T00:00",
            output_root=output_dir,
            override=True,
            break_on_error=False,
            station_configuration_root=station_configuration_root,
        )

        self.assertTrue(expected_compiled_output_file.exists())
        # There should only be a single output file since the first input file is empty
        self.assertEqual(1, len(list(expected_individual_output_dir.glob("*"))))
        self.assertTrue(expected_individual_output_file.exists())
        individual_data = expected_individual_output_file.read_bytes()
        compiled_data = expected_compiled_output_file.read_bytes()
        self.assertEqual(
            individual_data,
            compiled_data,
        )

    def test_get_bufr_where_period_does_not_exist(self):
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"
        input_file = input_dir / "THU_L2_hourly.csv"
        create_data_file(input_file, src_path=DATA_DIR.joinpath("tx_l3_test1.csv"))
        station_configuration_root = self.temp_dir / "station_configuration"
        station_configuration = get_station_configuration(
            stid="THU_L2", export_bufr=True
        )
        write_station_configuration_mapping(
            station_configurations={station_configuration.stid: station_configuration},
            configuration_root_dir=station_configuration_root,
        )

        create_bufr_files(
            input_files=[input_file],
            period_start="2025-12-06T00:00",
            period_end="2025-12-06T04:00",
            output_root=output_dir,
            override=True,
            break_on_error=True,
            station_configuration_root=station_configuration_root,
        )

        compiled_output_dir = output_dir / "compiled"
        individual_output_root = output_dir / "individual"
        self.assertTrue(compiled_output_dir.exists())
        self.assertTrue(individual_output_root.exists())
        expected_output_timestamps = [
            "20251206T0000",
            "20251206T0100",
            "20251206T0200",
            "20251206T0300",
            "20251206T0400",
        ]
        compiled_output_files = sorted(compiled_output_dir.glob("*.bufr"))
        expected_output_file_names = sorted(
            [
                f"geus_{timestamp_str}.bufr"
                for timestamp_str in expected_output_timestamps
            ]
        )
        self.assertListEqual(
            expected_output_file_names, [p.name for p in compiled_output_files]
        )
        for file in compiled_output_files:
            # The compiled bufr files should be empty
            self.assertEqual(0, file.stat().st_size)
        individual_output_dirs = sorted(individual_output_root.glob("*"))
        self.assertListEqual(
            expected_output_timestamps, [p.stem for p in individual_output_dirs]
        )
        for dir in individual_output_dirs:
            # There should be no bufr files in the individual directories
            self.assertEqual(0, len(list(dir.glob("*.bufr"))))
