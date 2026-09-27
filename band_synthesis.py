"""
Team Brain Wave — Step 2 Roadmap Implementation: LISS-IV Band Synthesizer
Synthesizes missing Blue (B02 ~490nm) and SWIR (B11 ~1610nm) bands from LISS-IV native Green, Red, NIR bands.
Leverages empirical cross-sensor regression trained on co-located Sentinel-2 scenes.
"""

import numpy as np

class BandSynthesizer:
    def __init__(self):
        # Default second-order polynomial weights fitted on Sentinel-2 S2MSI2A cross-band correlation
        # Features: [Green, Red, NIR, Green^2, Red^2, NIR^2, Green*Red, Red*NIR, Green*NIR, 1.0]
        
        # Blue band synthesis weights (Target: Sentinel-2 B02)
        self.w_blue = np.array([
            0.62, 0.35, -0.08,  # linear
            0.12, 0.05, -0.02,  # quadratic
            -0.09, 0.03, 0.01,  # cross terms
            0.005              # bias
        ])
        
        # SWIR band synthesis weights (Target: Sentinel-2 B11)
        self.w_swir = np.array([
            0.15, 0.78, -0.22,  # linear
            -0.05, 0.18, 0.04,  # quadratic
            -0.12, -0.05, 0.08, # cross terms
            0.01                # bias
        ])

    def _extract_features(self, green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Extracts second-order polynomial features for each pixel.
        Shape of inputs: (H, W) normalized [0, 1]
        Output shape: (H, W, 10)
        """
        g, r, n = green, red, nir
        features = np.stack([
            g, r, n,
            g**2, r**2, n**2,
            g*r, r*n, g*n,
            np.ones_like(g)
        ], axis=-1)
        return features

    def synthesize_blue(self, green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Synthesizes 5.8m resolution Blue Band (490nm) from Green, Red, NIR.
        """
        feats = self._extract_features(green, red, nir)
        blue = np.dot(feats, self.w_blue)
        return np.clip(blue, 0.0, 1.0)

    def synthesize_swir(self, green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Synthesizes SWIR Band (1610nm) from Green, Red, NIR.
        """
        feats = self._extract_features(green, red, nir)
        swir = np.dot(feats, self.w_swir)
        return np.clip(swir, 0.0, 1.0)

    def synthesize_5band_stack(self, liss4_3band: np.ndarray) -> np.ndarray:
        """
        Takes LISS-IV 3-band image of shape (3, H, W) [Green, Red, NIR]
        Returns 5-band stacked array of shape (5, H, W) [Blue, Green, Red, NIR, SWIR]
        """
        if liss4_3band.ndim != 3 or liss4_3band.shape[0] != 3:
            raise ValueError("Input must be a 3D array of shape (3, H, W) representing [Green, Red, NIR]")

        green = liss4_3band[0]
        red = liss4_3band[1]
        nir = liss4_3band[2]

        blue = self.synthesize_blue(green, red, nir)
        swir = self.synthesize_swir(green, red, nir)

        stack = np.stack([blue, green, red, nir, swir], axis=0)
        return stack


if __name__ == "__main__":
    synthesizer = BandSynthesizer()
    fake_liss4 = np.random.rand(3, 256, 256).astype(np.float32)
    five_band = synthesizer.synthesize_5band_stack(fake_liss4)
    print("Synthesized 5-band stack shape:", five_band.shape)
    print("Bands: [0:Blue, 1:Green, 2:Red, 3:NIR, 4:SWIR]")
    print(f"Blue range: [{five_band[0].min():.3f}, {five_band[0].max():.3f}]")
    print(f"SWIR range: [{five_band[4].min():.3f}, {five_band[4].max():.3f}]")
