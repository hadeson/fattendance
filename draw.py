import cv2

class DisplayDraw(object):
    def __init__(self, parent=None):
        self.indicator_text_setting = {
            'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
            'fontScale': 1.0,
            'color': (0, 255, 0),
            'thickness': 1,
            'lineType': cv2.LINE_AA
        }
        self.last_text = "Hello, I'm Miris"
        self.last_text_setting = self.indicator_text_setting

    def drawText(self, frame, text, text_setting):
        """
        "Look at camera" text and face bounding box
        """
        ts = text_setting
        screen_height, screen_width, _ = frame.shape
        text_color = ts['color']
        # if text == "Verification failed":
        #     text_color = (0, 0, 255) # red
        
        text_size = cv2.getTextSize(text, ts['fontFace'], ts['fontScale'], 
                                    ts['thickness'])
        # print(text_size, screen_width)
        text_position = (
            int((screen_width / 2) - (text_size[0][0] / 2)), 
            int(screen_height - 25)
        )
        # print(text_position)
        # print(ts)
        cv2.putText(
            frame, text, text_position, 
            ts['fontFace'], ts['fontScale'],
            text_color, 
            thickness=ts['thickness'],
            lineType=ts['lineType']
        )
        self.last_text = text
        self.last_text_setting = text_setting
        return frame

    def drawFailedText(self, frame):
        text = "Verification failed"
        ts = self.indicator_text_setting
        ts['color'] = (0, 0, 255) # red
        frame = self.drawText(frame, text, ts)
        return frame

    def drawDefaultText(self, frame):
        text = "Hello, I'm Miris"
        ts = self.indicator_text_setting
        ts['color'] = (255, 255, 255) # white
        frame = self.drawText(frame, text, ts)
        return frame

    def drawLACText(self, frame):
        text = "Please look at the camera below"
        ts = self.indicator_text_setting
        ts['color'] = (255, 255, 255) # white
        frame = self.drawText(frame, text, ts)
        return frame

    def drawSuccessText(self, frame, name):
        text = "Verification success, hello {}!".format(name)
        ts = self.indicator_text_setting
        ts['color'] = (0, 255, 0) # green
        frame = self.drawText(frame, text, ts)
        return frame

    def drawMCText(self, frame):
        text = "Please move closer"
        ts = self.indicator_text_setting
        ts['color'] = (255, 255, 255) # white
        frame = self.drawText(frame, text, ts)
        return frame

    def drawLastText(self, frame):
        return self.drawText(frame, self.last_text, self.last_text_setting)

