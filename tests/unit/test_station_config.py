from pathlib import Path
from unittest import TestCase
from tempfile import TemporaryDirectory

from pypromice.station_configuration import (
    StationConfiguration,
    load_station_configuration_mapping,
    write_station_configuration_mapping,
)
from tests.utilities import get_station_configuration


class StationConfigurationTestCase(TestCase):
    def test_read_toml(self):
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "UPE_L.toml"
            source_str = """
                stid = "UPE_L"
                station_site = "UPE_L"
                project = "Promice"
                station_type = "mobile"
                wmo_id = "04423"
                barometer_from_gps = -0.25
                anemometer_from_sonic_ranger = 0.4
                temperature_from_sonic_ranger = 0.0
                height_of_gps_from_station_ground = 0.9
                sonic_ranger_from_gps = 1.3
                export_bufr = true
                skipped_variables = []
                positions_update_timestamp_only = false
            """
            with source_path.open("w") as source_io:
                source_io.writelines(source_str)

            expected_configuration = StationConfiguration(
                stid="UPE_L",
                station_site="UPE_L",
                project="Promice",
                station_type="mobile",
                wmo_id="04423",
                barometer_from_gps=-0.25,
                anemometer_from_sonic_ranger=0.4,
                temperature_from_sonic_ranger=0.0,
                height_of_gps_from_station_ground=0.9,
                sonic_ranger_from_gps=1.3,
                export_bufr=True,
                comment=None,
                skipped_variables=[],
                positions_update_timestamp_only=False,
            )

            station_configuration = StationConfiguration.load_toml(source_path)
            self.assertEqual(
                expected_configuration,
                station_configuration,
            )

    def test_read_toml_with_unexpected_field(self):
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "UPE_L.toml"
            source_str = """
                stid = "UPE_L"
                station_site = "UPE_L"
                project = "Promice"
                station_type = "mobile"
                wmo_id = "04423"
                barometer_from_gps = -0.25
                anemometer_from_sonic_ranger = 0.4
                temperature_from_sonic_ranger = 0.0
                height_of_gps_from_station_ground = 0.9
                sonic_ranger_from_gps = 1.3
                export_bufr = true
                skipped_variables = []
                positions_update_timestamp_only = false
                an_unexpected_field = 42
            """
            with source_path.open("w") as source_io:
                source_io.writelines(source_str)

            expected_configuration = StationConfiguration(
                stid="UPE_L",
                station_site="UPE_L",
                project="Promice",
                station_type="mobile",
                wmo_id="04423",
                barometer_from_gps=-0.25,
                anemometer_from_sonic_ranger=0.4,
                temperature_from_sonic_ranger=0.0,
                height_of_gps_from_station_ground=0.9,
                sonic_ranger_from_gps=1.3,
                export_bufr=True,
                comment=None,
                skipped_variables=[],
                positions_update_timestamp_only=False,
            )

            with self.assertRaises(ValueError):
                StationConfiguration.load_toml(source_path)

            station_configuration = StationConfiguration.load_toml(source_path, skip_unexpected_fields=True)

            self.assertEqual(
                expected_configuration,
                station_configuration,
            )



    def test_write_read(self):
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "UPE_L.toml"
            src_station_config = StationConfiguration(
                stid="UPE_L",
                station_site="UPE_L",
                project="Promice",
                station_type="mobile",
                wmo_id="04423",
                barometer_from_gps=-0.25,
                anemometer_from_sonic_ranger=0.4,
                temperature_from_sonic_ranger=0.0,
                height_of_gps_from_station_ground=0.9,
                sonic_ranger_from_gps=1.3,
                export_bufr=True,
                comment=None,
                skipped_variables=[],
                positions_update_timestamp_only=False,
            )
            src_station_config.dump_toml(output_path)

            read_station_config = StationConfiguration.load_toml(output_path)
            self.assertEqual(
                src_station_config,
                read_station_config,
            )

    def test_read_station_config_mapping(self):
        with TemporaryDirectory() as temp_dir:
            station_config_root = Path(temp_dir) / "station_configurations"
            station_config_root.mkdir()
            source_mapping = {
                "UPE_L": get_station_configuration(stid="UPE_L"),
                "UPE_R": get_station_configuration(stid="UPE_R"),
            }
            for stid, station_config in source_mapping.items():
                station_config.dump_toml(station_config_root / f"{stid}.toml")

            read_mapping = load_station_configuration_mapping(station_config_root)
            self.assertDictEqual(
                source_mapping,
                read_mapping,
            )

    def test_write_station_config_mapping(self):
        with TemporaryDirectory() as temp_dir:
            station_config_root = Path(temp_dir) / "station_configurations"
            station_config_root.mkdir()
            source_mapping = {
                "UPE_L": get_station_configuration(stid="UPE_L"),
                "UPE_R": get_station_configuration(stid="UPE_R"),
            }

            write_station_configuration_mapping(source_mapping, station_config_root)

            read_mapping = load_station_configuration_mapping(station_config_root)
            self.assertDictEqual(
                source_mapping,
                read_mapping,
            )

    def test_read_station_config_mapping_empty(self):
        with TemporaryDirectory() as temp_dir:
            station_config_root = Path(temp_dir) / "station_configurations"
            station_config_root.mkdir()

            read_mapping = load_station_configuration_mapping(station_config_root)
            self.assertDictEqual(
                {},
                read_mapping,
            )

    def test_read_station_config_mapping_ingore_filenames(self):
        def test_read_station_config_mapping(self):
            with TemporaryDirectory() as temp_dir:
                station_config_root = Path(temp_dir) / "station_configurations"
                station_config_root.mkdir()
                station_config = get_station_configuration(stid="UPE_L")
                station_config.dump_toml(station_config_root / "a_custom_filename.toml")
                expected_station_config_mapping = {station_config.stid: station_config}

                read_mapping = load_station_configuration_mapping(station_config_root)
                self.assertDictEqual(
                    expected_station_config_mapping,
                    read_mapping,
                )
