# Experiment with Bidirectional Attention Flow Model

To run experiments with the [Resnet](https://arxiv.org/abs/1512.03385) model use [onnx_resnet.ipynb](../onnx_resnet.ipynb) notebook. Using this notebook we conducted experiments with the pre-trained ResNet classifier.

## Dataset

For performing this experiment, we will use some random contexts and queries.

## Prediction Results

### Simple Context

| Context | Query | Answer |
| --------| ----- | ------------------------------------------------ |
| A red Tesla is parked beside my house. | What color is the car? | 'red' |
| More human twins are being born now than ever before. | Who are born now? | Who are born now? |
| The first person convicted of speeding was going eight mph. | What speed was doing the first convicted person for speeding? | 'eight', 'mph' |
| The world wastes about 1 billion metric tons of food each year. | How much food we waste? | '1', 'billion', 'metric', 'tons' |
| Pineapple works as a natural meat tenderizer. | What is a meat tenderizer? | 'pineapple' |

### Complex Context

> Specifically, the hottest spot ever recorded on Earth is El Azizia, in Libya, where a temperature of 136 degrees Fahrenheit was recorded on Sept. 13, 1922. While hotter spots have likely occurred in other parts of the planet at other times, this is the most scorching temperature ever formally recorded by a weather station.

| Query | Answer |
| ----- | ------ |
| Where the hottest spot on Earth? | 'el', 'azizia' |
| Where is El Azizia? | 'libya' |
| What temperature has been detected? | '136', 'degrees', 'fahrenheit' |
| When this temperature has been recorded? | 'sept.', '13', ',', '1922' |
| By which station it was recorded? | 'weather', 'station' |
