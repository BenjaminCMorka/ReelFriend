import mongoose from 'mongoose';

const recommendationSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  movieIds: [{
    type: String,
    required: true
  }],
  explanations: [{
    type: String,
    default: ''
  }],
  generatedAt: {
    type: Date,
    default: Date.now
  },
  expiresAt: {
    type: Date,
    default: () => new Date(+new Date() + 24*60*60*1000) // 24 hours from now
  }
});

// remove unique index on user
// add TTL index on expiresAt
recommendationSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

export const Recommendation = mongoose.model('Recommendation', recommendationSchema);