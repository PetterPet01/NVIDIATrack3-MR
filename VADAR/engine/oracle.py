class CLEVROracle:
    def __init__(self):
        self.equivalence_classes = {
            "sphere": ["sphere", "ball"],
            "square": ["square", "cube", "block"],
            "cylinder": ["cylinder"],
            "large": ["large", "big"],
            "small": ["small", "tiny"],
            "metal": ["metallic", "metal", "shiny"],
            "rubber": ["rubber", "matte"],
        }
        self.possible_attributes = {
            "Shape": ["square", "sphere", "cylinder"],
            "Color": [
                "gray",
                "red",
                "blue",
                "green",
                "brown",
                "purple",
                "cyan",
                "yellow",
            ],
            "Relation": ["left", "right", "behind", "front"],
            "Size": ["small", "large"],
            "Material": ["rubber", "metal"],
        }
        self.large_box_scale_factor = 3
        self.depth_scale_factor = 150

    def locate(self, object_description, scene_json):
        matching_objects = self._find_objects(scene_json, object_description)
        pts = [
            (obj["pixel_coords"][0], obj["pixel_coords"][1]) for obj in matching_objects
        ]
        return pts

    def depth(self, x, y, scene_json):
        objects = scene_json["objects"]
        closest_obj = min(
            objects,
            key=lambda obj: (
                (obj["pixel_coords"][0] - x) ** 2 + (obj["pixel_coords"][1] - y) ** 2
            )
            ** 0.5,
        )
        return closest_obj["pixel_coords"][2]

    def same_object(self, x_1, y_1, x_2, y_2, scene_json):
        objects = scene_json["objects"]
        closest_obj_1 = min(
            objects,
            key=lambda obj: (
                (obj["pixel_coords"][0] - x_1) ** 2 + (obj["pixel_coords"][1] - y_1) ** 2
            )
            ** 0.5,
        )
        closest_obj_2 = min(
            objects,
            key=lambda obj: (
                (obj["pixel_coords"][0] - x_2) ** 2 + (obj["pixel_coords"][1] - y_2) ** 2
            )
            ** 0.5,
        )
        return closest_obj_1 == closest_obj_2

    def answer_question(self, x, y, question, scene_json):
        attribute = self._identify_attribute(question)

        objects = scene_json["objects"]

        closest_obj = min(
            objects,
            key=lambda obj: (
                (obj["pixel_coords"][0] - x) ** 2 + (obj["pixel_coords"][1] - y) ** 2
            )
            ** 0.5,
        )

        if attribute == "shape" and closest_obj["shape"] == "cube":
            return "square"

        return closest_obj[attribute]

    def _find_objects(self, scene_json, description):
        objects = scene_json["objects"]
        if description == "objects":
            return objects
        color, size, shape, material = self._parse_description(description)
        print(color, size, shape, material)
        matching_objects = self._filter_objects(objects, color, size, shape, material)
        return matching_objects

    def _parse_description(self, description):
        if description.endswith("s"):
            description = description[:-1]
        color, size, shape, material = None, None, None, None

        for word in description.split():
            word = self._map_to_equivalence_class(word)
            if word in self.possible_attributes["Color"]:
                color = word
            if word in self.possible_attributes["Size"]:
                size = word
            if word in self.possible_attributes["Shape"]:
                shape = word
            if word in self.possible_attributes["Material"]:
                material = word

        return color, size, shape, material

    def _map_to_equivalence_class(self, word):
        for key, synonyms in self.equivalence_classes.items():
            if word in synonyms:
                return key
        return word

    def _filter_objects(
        self, objects, color=None, size=None, shape=None, material=None
    ):
        filtered_objects = []
        for obj in objects:
            if color and obj["color"] != color:
                continue
            if size and obj["size"] != size:
                continue
            if shape and shape == "square" and obj["shape"] != "cube":
                continue
            if shape and shape != "square" and obj["shape"] != shape:
                continue
            if material and obj["material"] != material:
                continue
            filtered_objects.append(obj)
        return filtered_objects

    def _identify_attribute(self, question):
        question = question.lower()
        if "color" in question:
            return "color"
        elif "shape" in question:
            return "shape"
        elif "material" in question:
            return "material"
        elif "size" in question:
            return "size"
        else:
            raise ValueError("Question does not ask for a recognized attribute")
