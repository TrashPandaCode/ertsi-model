from dataset import ReverbRoomDataset
import os
import torch
from PIL import Image

class RoomAwareReverbDataset(ReverbRoomDataset):
    def __init__(self, data_dir, freqs=None, augment=False, include_room_id=True):
        # Call parent constructor with correct parameters
        super().__init__(data_dir, freqs=freqs, augment=augment)
        self.include_room_id = include_room_id
        
        # Create room ID mapping
        if include_room_id:
            self.room_to_id = self._create_room_mapping()
            
    def _create_room_mapping(self):
        room_names = set()
        for img_path, _ in self.entries:
            room_name = os.path.basename(os.path.dirname(img_path))
            room_names.add(room_name)
        
        return {room: idx for idx, room in enumerate(sorted(room_names))}
    
    def __getitem__(self, idx):
        img_path, rt60_values = self.entries[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        
        # Use the transform from parent class
        if self.augment and hasattr(self, 'augment_transform'):
            image = self.augment_transform(image)
        elif hasattr(self, 'base_transform'):
            image = self.base_transform(image)
        else:
            # Fallback to the general transform attribute
            image = self.transform(image)
        
        # Fix tensor creation warning
        if isinstance(rt60_values, torch.Tensor):
            rt60_tensor = rt60_values.clone().detach()
        else:
            rt60_tensor = torch.tensor(rt60_values, dtype=torch.float32)
        
        if self.include_room_id:
            room_name = os.path.basename(os.path.dirname(img_path))
            room_id = self.room_to_id[room_name]
            return image, rt60_tensor, torch.tensor(room_id, dtype=torch.long)
        
        return image, rt60_tensor