import re
from datetime import date, datetime
from google.cloud import storage
from pathlib import Path
from cropharvest.countries import BBox
from ipyleaflet import Map, basemaps, basemap_to_tiles, Rectangle
from ipywidgets import (
    Box,
    DatePicker,
    Dropdown,
    FloatText,
    FloatSlider,
    GridspecLayout,
    Layout,
    RadioButtons,
    Select,
    ToggleButtons,
)
from openmapflow.config import BucketNames
from openmapflow.labeled_dataset import bbox_from_path


class InferenceWidget:
    def __init__(
        self,
        available_models,
        lat: float = 7.72,
        lon: float = 1.18,
        margin: float = 0.02,
    ):
        self.bbox = BBox(
            min_lat=lat - margin,
            max_lat=lat + margin,
            min_lon=lon - margin,
            max_lon=lon + margin,
        )

        self.margin = margin

        self.existing_bboxes = self.get_available_bboxes()

        self.map = Map(
            layers=(basemap_to_tiles(basemaps.Esri.WorldStreetMap),),
            center=self.bbox.get_centre(in_radians=False),
            zoom=11,
        )
        self.map.add_layer(self.get_rectangle())

        self.model_picker = Dropdown(
            options=available_models, description="Model to use"
        )

        self.start_date = date(2020, 2, 1)
        self.end_date = date(2021, 2, 1)

    @staticmethod
    def get_available_bboxes():
        client = storage.Client()
        blobs = client.list_blobs(bucket_or_name=BucketNames.INFERENCE_TIFS)
        previous_matches = []
        available_bboxes = []
        long_regex = r".*min_lat=?\d*\.?\d*_min_lon=?\d*\.?\d*_max_lat=?\d*\.?\d*_max_lon=?\d*\.?\d*_dates=\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}.*?\/"
        blobs = client.list_blobs(bucket_or_name=BucketNames.INFERENCE_TIFS)
        for blob in blobs:
            match = re.search(long_regex, blob.name)
            if not match:
                continue
            p = match.group()
            if p not in previous_matches:
                previous_matches.append(p)
                available_bboxes.append(bbox_from_path(Path(p)))
        return available_bboxes

    def get_date_range(self):
        if self.gcs_path:
            pass
        else:
            return self.start_select

    def get_rectangle(self):
        return Rectangle(
            bounds=(
                (self.bbox.min_lat, self.bbox.min_lon),
                (self.bbox.max_lat, self.bbox.max_lon),
            )
        )

    def update_event(self, event):
        if event["name"] != "value":
            return
        key = event["owner"].description
        value = event["new"]
        self.gcs_path = ""
        if key == "Start date":
            self.start_date = value
        elif key == "End date":
            self.end_date = value
        elif key == "lat":
            self.bbox = BBox(
                min_lat=value - self.margin,
                max_lat=value + self.margin,
                min_lon=self.bbox.min_lon,
                max_lon=self.bbox.max_lon,
            )
        elif key == "lon":
            self.bbox = BBox(
                min_lat=self.bbox.min_lat,
                max_lat=self.bbox.max_lat,
                min_lon=value - self.margin,
                max_lon=value + self.margin,
            )
        elif key == "margin":
            lat, lon = self.bbox.get_centre(in_radians=False)
            self.bbox = BBox(
                min_lat=lat - value,
                max_lat=lat + value,
                min_lon=lon - value,
                max_lon=lon + value,
            )
            self.margin = value
        elif key in ["min_lat", "min_lon", "max_lat", "max_lon"]:
            setattr(self.bbox, key, value)
        else:
            for bbox in self.existing_bboxes:
                if bbox.name == value:
                    self.bbox = bbox
                    date_re = re.findall(r"\d{4}-\d{2}-\d{2}", value)
                    try:
                        self.start_date, self.end_date = [
                            datetime.strptime(d, "%Y-%m-%d").date() for d in date_re[:2]
                        ]
                    except:
                        print(
                            f"Could not parse dates from {value}, found {date_re}. Select a different bbox."
                        )
                    break

        self.map.substitute_layer(self.map.layers[-1], self.get_rectangle())
        self.map.center = self.bbox.get_centre(in_radians=False)

    def square_select(self):
        lat, lon = self.bbox.get_centre(in_radians=False)
        lat_input = FloatText(value=lat, description="lat")
        lon_input = FloatText(value=lon, description="lon")
        slider = FloatSlider(
            value=self.margin,
            min=0.01,
            max=3,
            step=0.01,
            description="margin",
            readout_format=".2f",
        )
        lat_input.observe(self.update_event)
        lon_input.observe(self.update_event)
        slider.observe(self.update_event)
        layout = Layout(flex_flow="column")
        return Box([Box([lat_input, lon_input]), slider], layout=layout)

    def rectangle_select(self):
        inputs = {}
        for k in ["min_lat", "max_lat", "min_lon", "max_lon"]:
            inputs[k] = FloatText(value=getattr(self.bbox, k), description=k)
            inputs[k].observe(self.update_event)
        grid = GridspecLayout(3, 3)
        for i in range(3):
            for j in range(3):
                grid[i, j] = FloatText(value=0.0)
                grid[i, j].layout.visibility = "hidden"
        grid[0, 1] = inputs["max_lat"]
        grid[1, 0] = inputs["min_lon"]
        grid[1, 2] = inputs["max_lon"]
        grid[2, 1] = inputs["min_lat"]
        return Box([grid])

    def existing_bbox_select(self):
        select = Select(options=[bbox.name for bbox in self.existing_bboxes])
        select.observe(self.update_event)
        return select

    def bbox_select(self):
        square_select = self.square_select()
        rectangle_select = self.rectangle_select()
        cached_display = rectangle_select.layout.display
        radio_buttons = RadioButtons(
            description="Bbox for map", options=["Square", "Rectangle"]
        )

        def change_visibility(event):
            try:
                i = event["new"]["index"]
            except:
                return
            rectangle_select.layout.display = cached_display if i == 1 else "none"
            square_select.layout.display = "block" if i == 0 else "none"

        rectangle_select.layout.display = "none"

        radio_buttons.observe(change_visibility)
        return Box(
            [radio_buttons, square_select, rectangle_select],
            layout=Layout(flex_flow="column"),
        )

    def new_data_select(self):
        start_select = DatePicker(description="Start date", value=self.start_date)
        end_select = DatePicker(description="End date", value=self.end_date)
        start_select.observe(self.update_event)
        end_select.observe(self.update_event)
        return Box([start_select, end_select, self.bbox_select()])

    def ui(self):
        existing_data_available = len(self.existing_bboxes) > 0
        new_data_select = self.new_data_select()

        children = [self.model_picker]
        if existing_data_available:
            toggle = ToggleButtons(
                options=["Available data", "New data"], description="Data to use"
            )
            existing_data_select = self.existing_bbox_select()
            children += [toggle, existing_data_select, new_data_select]

            def change_visibility(event):
                try:
                    i = event["new"]["index"]
                except:
                    return
                existing_data_select.layout.display = "block" if i == 0 else "none"
                new_data_select.layout.display = "block" if i == 1 else "none"
                if i == 0:
                    self.update_event(
                        {
                            "name": "value",
                            "owner": existing_data_select,
                            "new": self.existing_bboxes[0].name,
                        }
                    )

            change_visibility({"new": {"index": 0}})
            toggle.observe(change_visibility)

        else:
            children += [new_data_select]

        return Box(children + [self.map], layout=Layout(flex_flow="column"))
