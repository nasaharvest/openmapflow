import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import partial

from openmapflow.bbox import BBox

try:
    import pyproj
    import shapely.ops as ops
    from ipyleaflet import Map, Rectangle, basemap_to_tiles, basemaps
    from ipywidgets import (
        HTML,
        Box,
        DatePicker,
        Dropdown,
        FloatText,
        Layout,
        RadioButtons,
        Select,
        ToggleButtons,
        VBox,
    )
    from shapely.geometry.polygon import Polygon
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "ipyleaflet, pyproj, shapely, or ipywidgets not found, please install with"
        + "`pip install ipyleaflet pyproj shapely ipywidgets`"
    )


bbox_keys = ["min_lat", "min_lon", "max_lat", "max_lon"]


@dataclass
class InferenceBBox(BBox):
    def __post_init__(self):
        super().__post_init__()
        self.area = self.get_area_km2()
        self.center = self.get_centre(in_radians=False)

    @classmethod
    def from_bbox(cls, bbox: BBox):
        return cls(bbox.min_lat, bbox.max_lat, bbox.min_lon, bbox.max_lon, bbox.name)

    def get_area_km2(self) -> float:
        polygon = Polygon(
            [
                [self.min_lon, self.min_lat],
                [self.min_lon, self.max_lat],
                [self.max_lon, self.max_lat],
                [self.max_lon, self.min_lat],
            ]
        )
        polygon = ops.transform(
            partial(
                pyproj.transform,
                pyproj.Proj(init="EPSG:4326"),
                pyproj.Proj(
                    proj="aea", lat_1=polygon.bounds[1], lat_2=polygon.bounds[3]
                ),
            ),
            polygon,
        )
        return polygon.area * 1e-6  # Convert from m2 to km2

    def get_leaflet_rectangle(self):
        return Rectangle(
            bounds=((self.min_lat, self.min_lon), (self.max_lat, self.max_lon))
        )

    def get_time_estimate(self):
        # TODO: with_earth_engine=True parameter
        # Earth Engine Export
        # margin 0.01 -> 1 min
        # margin 0.02 -> 3 mins
        # margin 0.03 -> 9 mins
        # margin 0.05 -> 10 mins

        # Inference
        # Start: 10:02:15
        # Done: 10:02:44
        pass


def create_coords_widget(bbox, margin, update_bbox):
    coord_widgets = {}
    for k in bbox_keys + ["lat", "lon", "margin"]:
        if k == "lat":
            value = bbox.center[0]
        elif k == "lon":
            value = bbox.center[1]
        elif k == "margin":
            value = margin
        else:
            value = getattr(bbox, k)
        coord_widgets[k] = FloatText(value=round(value, 3), description=k, step=0.01)
        coord_widgets[k].observe(update_bbox)
    return coord_widgets


def create_new_bbox_widget(get_bbox, coord_widgets):
    square_widget = VBox(
        [coord_widgets["lat"], coord_widgets["lon"], coord_widgets["margin"]]
    )
    lon_coords = Box([coord_widgets["min_lon"], coord_widgets["max_lon"]])
    all_coords = [coord_widgets["max_lat"], lon_coords, coord_widgets["min_lat"]]
    rectangle_widget = Box([VBox(all_coords, layout=Layout(align_items="center"))])
    cached_display = rectangle_widget.layout.display
    rectangle_widget.layout.display = "none"
    toggle = ToggleButtons(options=["Square bbox", "Rectangle bbox"])

    def change_visibility(event):
        try:
            i = event["new"]["index"]
        except Exception:
            return
        if i == 0:
            square_widget.layout.display = "block"
            rectangle_widget.layout.display = "none"
            bbox = get_bbox()
            coord_widgets["lat"].value = round(bbox.center[0], 3)
            coord_widgets["lon"].value = round(bbox.center[1], 3)
        elif i == 1:
            square_widget.layout.display = "none"
            rectangle_widget.layout.display = cached_display
            bbox = get_bbox()
            for k in bbox_keys:
                coord_widgets[k].value = round(getattr(bbox, k), 3)

    toggle.observe(change_visibility)
    return VBox([toggle, square_widget, rectangle_widget])


def create_available_bbox_widget(available_bboxes, update_event):
    options = [bbox.name for bbox in available_bboxes]
    select = Select(options=options, description="On Google Cloud")
    select.observe(update_event)
    return select


class InferenceWidget:
    def __init__(
        self,
        available_models,
        available_bboxes=[],
        lat: float = 7.72,
        lon: float = 1.18,
        margin: float = 0.02,
        start_date=date(2020, 2, 1),
        end_date=date(2021, 2, 1),
        verbose=False,
    ):
        self.verbose = verbose
        self.bbox = InferenceBBox(
            min_lat=lat - margin,
            max_lat=lat + margin,
            min_lon=lon - margin,
            max_lon=lon + margin,
        )
        self.available_bboxes = available_bboxes

        # -----------------------------------------------------------------------
        # Initialize all widgets
        # -----------------------------------------------------------------------
        layers = (
            basemap_to_tiles(basemaps.Esri.WorldStreetMap),
            self.bbox.get_leaflet_rectangle(),
        )
        self.map = Map(layers=layers, center=self.bbox.center, zoom=11)
        self.coord_widgets = create_coords_widget(self.bbox, margin, self.update_bbox)
        self.new_bbox_widget = create_new_bbox_widget(
            lambda: self.bbox, self.coord_widgets
        )
        self.available_bbox_widget = create_available_bbox_widget(
            available_bboxes, self.update_bbox
        )

        # Simple custom widgets
        self.model_picker = Dropdown(
            options=available_models, description="Model to use"
        )
        self.start_widget = DatePicker(description="Start date", value=start_date)
        self.end_widget = DatePicker(description="End date", value=end_date)
        for widget in [self.model_picker, self.start_widget, self.end_widget]:
            widget.observe(self.update_map_key)

        self.check_key_widget = RadioButtons(
            options=["Check existing progress", "Create new map"],
            layout=Layout(display="none"),
        )
        self.check_key_widget.observe(self.update_map_key)
        self.map_key_HTML = HTML(self.get_map_key_HTML())
        self.estimates_HTML = HTML(self.get_estimates_HTML())
        self.warning_HTML = HTML("", style={"color": "red"})

    def are_tifs_in_right_spot(self, map_key):
        return any(map_key in b.name for b in self.available_bboxes)

    def get_map_key(self):
        version = self.bbox.get_identifier(
            str(self.start_widget.value), str(self.end_widget.value)
        )
        map_key = f"{self.model_picker.value}/{version}"
        if self.are_tifs_in_right_spot(map_key):
            self.check_key_widget.layout.display = "block"
        else:
            self.check_key_widget.layout.display = "none"
            self.check_key_widget.value = "Check existing progress"

        if self.check_key_widget.value == "Check existing progress":
            return map_key

        try:
            new_version = (
                int(re.search(r"_v\d*", map_key).group().replace("_v", "")) + 1
            )
        except Exception:
            new_version = 1
        return f"{map_key}_v{new_version}"

    def get_map_key_HTML(self):
        return f"<b>Map key:</b> {self.get_map_key()}"

    def get_config_as_dict(self):
        map_key = self.get_map_key()
        return {
            "bbox": self.bbox,
            "start_date": self.start_widget.value,
            "end_date": self.end_widget.value,
            "tifs_in_gcloud": self.bbox.name,
            "tifs_in_right_spot": self.are_tifs_in_right_spot(map_key),
            "map_key": map_key,
        }

    def get_estimates_HTML(self) -> str:
        return f"""
          <div style='padding-left:1em'>
          <h3>Estimates</h3>
          <b>Area:</b> {self.bbox.get_area_km2():,.1f} km² <br>
          <b>Time:</b> Coming soon. <br>
          <b>Cost:</b> Coming soon.
          <div/>
        """

    def get_warning_HTML(self, description):
        return f"<p style='color:red'>{description}</p>"

    def update_map_key(self, event):
        if event["name"] != "value":
            return
        start_date = self.start_widget.value
        end_date = self.end_widget.value
        model_name = self.model_picker.value
        if str(start_date.year) not in model_name:
            self.warning_HTML.value = self.get_warning_HTML(
                f"Start year: {start_date.year} not in model name: {model_name}."
            )
        elif start_date.month != end_date.month:
            self.warning_HTML.value = self.get_warning_HTML(
                f"Start month {start_date.month} and end month {end_date.month} should be the same."
            )
        elif start_date.year + 1 != end_date.year:
            self.warning_HTML.value = self.get_warning_HTML(
                f"Start year {start_date.year} should be one less than end year {end_date.year}"
            )
        else:
            self.warning_HTML.value = ""
        self.map_key_HTML.value = self.get_map_key_HTML()

    def update_bbox(self, event):
        if event["name"] != "value":
            return
        key = event["owner"].description
        value = event["new"]
        if key == "lat":
            self.bbox = InferenceBBox(
                min_lat=value - self.coord_widgets["margin"].value,
                max_lat=value + self.coord_widgets["margin"].value,
                min_lon=self.bbox.min_lon,
                max_lon=self.bbox.max_lon,
            )
        elif key == "lon":
            self.bbox = InferenceBBox(
                min_lat=self.bbox.min_lat,
                max_lat=self.bbox.max_lat,
                min_lon=value - self.coord_widgets["margin"].value,
                max_lon=value + self.coord_widgets["margin"].value,
            )
        elif key == "margin":
            lat, lon = self.bbox.center
            self.bbox = InferenceBBox(
                min_lat=lat - value,
                max_lat=lat + value,
                min_lon=lon - value,
                max_lon=lon + value,
            )
        elif key in bbox_keys:
            kwargs = {k: v for k, v in self.bbox.__dict__.items() if k in bbox_keys}
            kwargs[key] = value
            try:
                self.bbox = InferenceBBox(**kwargs)
                self.warning_HTML.value = ""
            except ValueError as e:
                self.warning_HTML.value = self.get_warning_HTML(str(e))

        elif key == "On Google Cloud":
            try:
                bbox = next(b for b in self.available_bboxes if b.name == value)
                self.bbox = InferenceBBox.from_bbox(bbox)
                ds = re.findall(r"\d{4}-\d{2}-\d{2}", value)
                self.start_widget.value = datetime.strptime(ds[0], "%Y-%m-%d").date()
                self.end_widget.value = datetime.strptime(ds[1], "%Y-%m-%d").date()
                self.warning_HTML.value = ""
            except Exception as e:
                self.warning_HTML.value = self.get_warning_HTML(str(e))

        if self.verbose:
            print(f"Updated bbox from key: {key}")
            print(self.bbox)

        self.map_key_HTML.value = self.get_map_key_HTML()
        self.estimates_HTML.value = self.get_estimates_HTML()
        self.map.center = self.bbox.center
        self.map.substitute_layer(
            self.map.layers[-1], self.bbox.get_leaflet_rectangle()
        )

    def change_new_vs_available(self, event):
        try:
            i = event["new"]["index"]
        except Exception:
            return
        if i == 0:
            self.available_bbox_widget.layout.display = "block"
            self.new_bbox_widget.layout.display = "none"
            self.start_widget.disabled = True
            self.end_widget.disabled = True
            self.update_bbox(
                {
                    "name": "value",
                    "owner": self.available_bbox_widget,
                    "new": self.available_bboxes[0].name,
                }
            )
        elif i == 1:
            self.available_bbox_widget.layout.display = "none"
            self.new_bbox_widget.layout.display = "block"
            self.start_widget.disabled = False
            self.end_widget.disabled = False

    def ui(self):
        config_title = HTML("<h3>Select model and specify region of interest</h3>")
        dates = VBox([self.start_widget, self.end_widget])
        if len(self.available_bboxes) == 0:
            configuration = VBox([config_title, Box([self.model_picker, dates])])
            return VBox(
                [
                    Box([configuration, self.estimates_HTML]),
                    self.new_bbox_widget,
                    self.map_key_HTML,
                    self.warning_HTML,
                    self.map,
                ]
            )
        toggle = ToggleButtons(options=["Available regions", "New regions"])
        toggle.observe(self.change_new_vs_available)
        model_and_toggle = VBox([self.model_picker, toggle])
        configuration = VBox([config_title, Box([model_and_toggle, dates])])
        self.change_new_vs_available({"new": {"index": 0}})
        return VBox(
            [
                Box([configuration, self.estimates_HTML]),
                self.available_bbox_widget,
                self.new_bbox_widget,
                self.map_key_HTML,
                self.check_key_widget,
                self.warning_HTML,
                self.map,
            ]
        )
